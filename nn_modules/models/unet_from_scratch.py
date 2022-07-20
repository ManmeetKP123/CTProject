from re import X
import torch 
import torch.nn as nn 


class SimpleConvolution(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channel, output_channel, (3, 3))
        self.conv2 = nn.Conv3d(output_channel, output_channel, (3, 3))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x
    
class DownConvolution(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channel, output_channel, (3, 3))
        self.conv2 = nn.Conv3d(output_channel, output_channel, (3, 3))
        self.maxpool = nn.MaxPool3d(2, 2)
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout3d(0.4)
    
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.Relu(x)
        x = self.dropout(x)

        return x


class UpConvolution(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channel, output_channel, (3, 3))
        self.conv2 = nn.Conv3d(output_channel, output_channel, (3, 3))
        self.convTranspose = nn.ConvTranspose3d(output_channel, output_channel//2, (2, 2), (2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d(0.4)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.convTranspose(x)
    
        return x 

class LastConvolution(nn.Module):
    def __init__(self, input_channels, output_channels, num_classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, (3, 3))
        self.conv2 = nn.Conv3d(output_channels, output_channels, (3, 3))
        self.conv3 = nn.Conv3d(output_channels, num_classes, (1,1))
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout3d(0.4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.conv3(x)

        return x 

def crop_img(source_tensor, target_tensor):
    source_tensor_size = source_tensor.size()[2]
    target_tensor_size = target_tensor.size()[2]
    diff = (source_tensor_size - target_tensor_size) // 2

    #tensors in our case are 5d because of 3d data
    return source_tensor[:,:, diff:-diff, diff:-diff, diff:-diff]


class UNet(nn.Module):
    def __init__(self, input_channels, num_classes) -> None:
        super().__init__()
        self.simpleConv = SimpleConvolution(input_channels, 64)
        self.downBlock1 = DownConvolution(64, 128)
        self.downBlock2 = DownConvolution(128, 256)
        self.downBlock3 = DownConvolution(256, 512)
        self.midMaxPool = nn.MaxPool3d(2, 2)
        self.bridge = UpConvolution(512, 1024)
        self.upBlock1 = UpConvolution(1024,  512)
        self.upBlock2 = UpConvolution(512, 256)
        self.upBlock3 = UpConvolution(256, 128)
        self.lastConv = LastConvolution(128, 64, num_classes)

    
    def forward(self, x):
        x_1 = self.simpleConv(x)
        x_2 = self.downBlock1(x_1)
        x_3 = self.downBlock2(x_2)
        x_4 = self.downBlock3(x_3)
        x_5 = self.midMaxPool(x_4)

        x_6 = self.bridge(x_5)
        crop_x_4 = crop_img(x_4, x_6)
        concat_x_4_6 = torch.cat((crop_x_4, x_6), 1)

        x_7 = self.upBlock1(concat_x_4_6)
        crop_x_3 = crop_img(x_3, x_7)
        concat_x_3_7 = torch.cat((crop_x_3, x_7), 1)

        x_8 = self.upBlock2(concat_x_3_7)
        crop_x_2 = crop_img(x_2, x_8)
        concat_x_2_8 = torch.cat((crop_x_2, x_8), 1)
        
        x_9 = self.upBlock3(concat_x_2_8)
        crop_x_1 = crop_img(x_1, x_9)
        concat_x_1_9 = torch.cat((crop_x_1, x_9), 1)

        out = self.lastConv(concat_x_1_9)

        return out


