"""
unet_model.py - U-Net Architecture for Image Colorization
---------------------------------------------------------
Input:  1 channel (Lightness 'L' from CIELAB color space, normalized to [0, 1] or [-1, 1])
Output: 2 channels ('a' and 'b' color opponent channels, bounded in [-1, 1] via Tanh)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Standard double convolution block:
    [Conv2d -> BatchNorm -> LeakyReLU -> Conv2d -> BatchNorm -> LeakyReLU]
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    """
    Upscaling block that uses Bilinear Upsampling + Conv (to prevent checkerboard artifacts)
    and concatenates skip connections from the encoder.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 2x spatial upsampling
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # in_channels of double_conv is out_channels (from up) + out_channels (from skip connection)
        self.double_conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv_reduce(x)

        # Handle any possible size mismatch due to odd dimensions
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        # Concatenate skip connection along channel dimension
        x = torch.cat([skip, x], dim=1)
        return self.double_conv(x)


class ColorizationUNet(nn.Module):
    """
    5-stage U-Net tailored specifically for CIELAB Image Colorization.
    - Deep feature extraction with skip connections to preserve fine textural boundaries.
    - Tanh activation ensures predicted (a, b) channels are bounded to [-1.0, 1.0].
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 2, base_filters: int = 64):
        super().__init__()

        # Encoder (Contracting Path)
        self.enc1 = DoubleConv(in_channels, base_filters)          # 64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(base_filters, base_filters * 2)     # 128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(base_filters * 2, base_filters * 4) # 256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(base_filters * 4, base_filters * 8) # 512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (Latent Representation)
        self.bottleneck = nn.Sequential(
            DoubleConv(base_filters * 8, base_filters * 16),        # 1024
            nn.Dropout2d(p=0.3)
        )

        # Decoder (Expanding Path with Skip Connections)
        self.up4 = UpBlock(base_filters * 16, base_filters * 8)    # 1024 -> 512
        self.up3 = UpBlock(base_filters * 8, base_filters * 4)     # 512 -> 256
        self.up2 = UpBlock(base_filters * 4, base_filters * 2)     # 256 -> 128
        self.up1 = UpBlock(base_filters * 2, base_filters)         # 128 -> 64

        # Final Color Head: maps 64 feature maps to 2 channels (a*, b*)
        self.color_head = nn.Sequential(
            nn.Conv2d(base_filters, base_filters // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters // 2, out_channels, kernel_size=1),
            nn.Tanh()  # Outputs bounded strictly between -1.0 and +1.0
        )

    def forward(self, L: torch.Tensor) -> torch.Tensor:
        """
        Args:
            L (torch.Tensor): Lightness channel tensor [Batch, 1, Height, Width]
        Returns:
            torch.Tensor: Predicted (a*, b*) chrominance channels [Batch, 2, Height, Width]
        """
        # Encoder passes
        s1 = self.enc1(L)
        p1 = self.pool1(s1)

        s2 = self.enc2(p1)
        p2 = self.pool2(s2)

        s3 = self.enc3(p2)
        p3 = self.pool3(s3)

        s4 = self.enc4(p3)
        p4 = self.pool4(s4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder passes with skip connections
        d4 = self.up4(b, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        # Color channels output
        ab = self.color_head(d1)
        return ab


if __name__ == "__main__":
    # Sanity verification of dimensions and forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorizationUNet(in_channels=1, out_channels=2).to(device)
    dummy_L = torch.randn(2, 1, 256, 256, device=device)
    out_ab = model(dummy_L)
    print("ColorizationUNet instantiated successfully.")
    print("Input L shape :", dummy_L.shape)
    print("Output AB shape:", out_ab.shape)
    print("Output range  :", out_ab.min().item(), "to", out_ab.max().item())
