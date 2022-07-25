import torch 
import torch.nn as nn
import torch.nn.functional as F 

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1), 
            nn.BatchNorm3d(in_ch), 
            nn.ReLU(inplace=True), 
            nn.Conv3d(in_ch, out_ch, 3, padding = 1), 
            nn.BatchNorm3d(out_ch), 
            nn.ReLU(inplace= True)
        )
    
    def forward(self, x):
        x = self.conv(x)

        return x 

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch // 2, 3, padding=1), 
            nn.BatchNorm3d(out_ch // 2), 
            nn.ReLU(inplace=True), 
            nn.Conv3d(out_ch // 2, out_ch, 3, padding = 1), 
            nn.BatchNorm3d(out_ch), 
            nn.ReLU(inplace= True)
        )
    
    def forward(self, x):
        x = self.conv(x)

        return x 

class Down(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2), 
            DoubleConv(in_ch, out_ch)
        )
    
    def forward(self, x):
        x = self.mpconv(x)
        
        return x 

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear = True) -> None:
        super().__init__()
        
        if trilinear: 
            self.up =  nn.Upsample(scale_factor = 2, mode="trilinear", align_corners = True)
        else: 
            self.up = nn.ConvTranspose3d(in_ch//2, in_ch // 2, 2, stride = 2)
        
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        print("dim of upsampled x1 " + str(x1.size()))
        print("dim of incoming x2 " + str(x2.size()))

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]
        print(diffY)
        print(diffX)
        print(diffZ)

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2))
        x = torch.cat([x2, x1], dim=1)
        print("dim after concatenation " + str(x.size()))
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        
    def forward(self, x):
        x = self.conv(x)
        return x 

class UNet(nn.Module): 
    def __init__(self, in_channels, classes) -> None:
        super().__init__()
        self.n_channels = in_channels 
        self.n_classes = classes 

        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 =  Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 =  Up(128, 64)
        self.outc = OutConv(64, classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        print("dim of x4 " + str(x4.size()))
        x5 = self.down4(x4)
        print("dim of x5 " + str(x5.size()))

        x = self.up1(x5, x4)
        print("dim of x " + str(x.size()))
        x = self.up2(x, x3)
        print("dim of x " + str(x.size()))
        x = self.up3(x, x2)
        print("dim of x " + str(x.size()))
        x = self.up4(x, x1)
        print("dim of x " + str(x.size()))
        x = self.outc(x)
        print("dim of x " + str(x.size()))

        return x