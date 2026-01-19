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
        return super().forward(x[:, [0], :, :])
    

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
    

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=8, expansion=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)

        hidden = int(channels * expansion)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTUNetDown(nn.Module):
    def __init__(self, in_channels, patch_size=4, num_heads=8):
        super().__init__()
        out_channels = in_channels * 2
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

        self.transformer = TransformerBlock(
            channels=out_channels,
            num_heads=num_heads
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.proj(x)

        B, C, H, W = x.shape
        p = self.patch_size

        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, -1, C)

        x = self.transformer(x)

        x = x.reshape(B, H // p, W // p, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, H, W)

        return x


class ViTUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_conv = DoubleConv(in_channels, 64)
        self.down1 = Down(64)
        self.down2 = Down(128)
        self.down3 = ViTUNetDown(256)
        self.down4 = ViTUNetDown(512)
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