from re import X
import torch 
import torch.nn as nn 


class SimpleConvolution(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channel, output_channel, kernel_size=3, stride = 1, padding=1)
        self.conv2 = nn.Conv3d(output_channel, output_channel, kernel_size=3, stride = 1, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        print("first line of forward")
        print("printing the type " + str(type(x)))
        x = self.conv1(x)
        print("second line of forward")
        x = self.relu(x)
        print("thrd line")
        x = self.conv2(x)
        print("fourth line")
        x = self.relu(x)
        print("fifth line")
        x = self.dropout(x)

        return x
    
class DownConvolution(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channel, output_channel, kernel_size=3, stride = 1, padding=1)
        self.conv2 = nn.Conv3d(output_channel, output_channel, kernel_size = 3, stride = 1, padding=1)
        self.maxpool = nn.MaxPool3d(2)
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout3d(0.2)
    
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
        self.conv1 = nn.Conv3d(input_channel, output_channel, kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv3d(output_channel, output_channel, kernel_size = 2, stride = 2)
        self.upsample = nn.Upsample(scale_factor = 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d(0.2)
    
    def forward(self, x):
        x = self.upsample(x)
        print('size of the upsampled inpt ' + str(x.size()))
        x = self.conv1(x)
        x = self.relu(x)
        print('size after conv 1 ' + str(x.size()))
        x = self.conv2(x)
        x = self.relu(x)
        print("size after conv 2 " + str(x.size()))
        x = self.dropout(x)
    
        return x 

class LastConvolution(nn.Module):
    def __init__(self, input_channels, output_channels, num_classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=3, stride = 1)
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=3, stride = 1)
        self.conv3 = nn.Conv3d(output_channels, num_classes, kernel_size=1, stride = 1)
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout3d(0.2)

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
        print(type(input_channels))
        print(type(64))
        self.simpleConv = SimpleConvolution(input_channels, 64)
        self.downBlock1 = DownConvolution(64, 128)
        self.downBlock2 = DownConvolution(128, 256)
        self.downBlock3 = DownConvolution(256, 512)
        self.downBlock4 = DownConvolution(512, 1024)
        self.bridge = nn.Sequential(nn.Conv3d(1024, 1024, kernel_size = 3, stride = 1), 
                                    nn.Conv3d(1024, 1024, kernel_size = 3, stride = 1))
        self.upBlock1 = UpConvolution(1024,  512)
        self.upBlock2 = UpConvolution(512, 256)
        self.upBlock3 = UpConvolution(256, 128)
        self.lastConv = LastConvolution(128, 64, num_classes)

        print("initialization successful")

    
    def forward(self, x):
        x_1 = self.simpleConv(x)
        print(x_1.size())
        x_2 = self.downBlock1(x_1)
        print(x_2.size())
        x_3 = self.downBlock2(x_2)
        print(x_3.size())
        x_4 = self.downBlock3(x_3)
        print(x_4.size())
        x_5 = self.downBlock4(x_4)
        print(x_5.size())
        # x_5 = self.midMaxPool(x_4)
        # print(x_5.size())

        x_6 = self.bridge(x_5)
        print("dimension of x_6 " +str(x_6.size()) )
        crop_x_4 = crop_img(x_4, x_6)
        print("dimension of cropped x_4 " + str(crop_x_4.size()))
        concat_x_4_6 = torch.cat((crop_x_4, x_6), 1)
        print("dimension of concatenated x_4_6 " + str(concat_x_4_6.size()))

        x_7 = self.upBlock2(concat_x_4_6)
        print("dimension of x_7 " + str(x_7.size()))
        crop_x_3 = crop_img(x_3, x_7)
        concat_x_3_7 = torch.cat((crop_x_3, x_7), 1)
        print("dimension after the concatenation 2 " + str(concat_x_3_7.size()))

        x_8 = self.upBlock3(concat_x_3_7)
        crop_x_2 = crop_img(x_2, x_8)
        concat_x_2_8 = torch.cat((crop_x_2, x_8), 1)
        print("dimension after the concatenation 3 " + str(concat_x_2_8.size()))
        
        x_9 = self.upBlock3(concat_x_2_8)
        crop_x_1 = crop_img(x_1, x_9)
        concat_x_1_9 = torch.cat((crop_x_1, x_9), 1)
        print("dimension after the concatenation 4 " + str(concat_x_1_9.size()))

        out = self.lastConv(concat_x_1_9)

        return out