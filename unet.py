import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Pure UNet
"""

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = 2 * in_channels
        self.down = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x):
        return self.conv(self.down(x))


class Up(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError("in_channels must be divisible by 2")
        out_channels = in_channels // 2
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dx = x2.shape[2] - x1.shape[2]
        dy = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, (dx//2, dx-dx//2, dy//2, dy-dy//2))
        x1 = torch.cat([x1, x2], dim=1)
        return self.conv(x1)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_conv = DoubleConv(in_channels, 64)
        self.down1 = Down(64)
        self.down2 = Down(128)
        self.down3 = Down(256)
        self.down4 = Down(512)
        self.up1 = Up(1024)
        self.up2 = Up(512)
        self.up3 = Up(256)
        self.up4 = Up(128)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)
    

class UNetVV(UNet):
    def __init__(self, in_channels, num_classes):
        super().__init__(1, num_classes)
    def forward(self, x):
        return super().forward(x[:, 0, :, :])
    

class UNetWithoutDEM(UNet):
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels-1, num_classes)
    def forward(self, x):
        return super().forward(x[:, :-1, :, :])
    

class UNetWithDerivedFeatures(UNet):
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels*3, num_classes)
    def forward(self, x):
        B, C, H, W = x.shape
        std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        mean = mean.expand(B, C, H, W)
        std = std.expand(B, C, H, W)
        x = torch.cat([x, mean, std], dim=1)
        return super().forward(x)
    

class UNetWithDerivedFeaturesWithoutDEM(UNet):
    def __init__(self, in_channels, num_classes):
        super().__init__((in_channels-1)*3, num_classes)
    def forward(self, x):
        x = x[:, :-1, :, :]
        B, C, H, W = x.shape
        std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        mean = mean.expand(B, C, H, W)
        std = std.expand(B, C, H, W)
        x = torch.cat([x, mean, std], dim=1)
        return super().forward(x)


class UNet6(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_conv = DoubleConv(in_channels, 16)
        self.down1 = Down(16)
        self.down2 = Down(32)
        self.down3 = Down(64)
        self.down4 = Down(128)
        self.down5 = Down(256)
        self.down6 = Down(512)
        self.up1 = Up(1024)
        self.up2 = Up(512)
        self.up3 = Up(256)
        self.up4 = Up(128)
        self.up5 = Up(64)
        self.up6 = Up(32)
        self.out_conv = nn.Conv2d(16, num_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.down6(x6)
        x = self.up1(x, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        return self.out_conv(x)
    

"""
UNet with ConvNext blocks
"""

class ConvNextBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(1e-6 * torch.ones(channels))

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)   # NCHW → NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)   # NHWC → NCHW
        return x + residual


class ConvNextStage(nn.Module):
    def __init__(self, channels, num_blocks=4) -> None:
        super().__init__()
        self.down = nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2)
        self.stage = nn.Sequential(
            *(ConvNextBlock(channels * 2) for _ in range(num_blocks))
        )
    def forward(self, x):
        x = self.down(x)
        x = self.stage(x)
        return x


class ConvNextUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_conv = DoubleConv(in_channels, 64)
        self.down1 = ConvNextStage(64, 1)
        self.down2 = ConvNextStage(128, 1)
        self.down3 = ConvNextStage(256, 1)
        self.down4 = ConvNextStage(512, 4)
        self.up1 = Up(1024)
        self.up2 = Up(512)
        self.up3 = Up(256)
        self.up4 = Up(128)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)