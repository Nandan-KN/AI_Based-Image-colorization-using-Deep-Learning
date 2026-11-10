import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# Convolution block
# ============================================================

class ConvNormAct(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1
    ):
        super().__init__()

        groups = min(8, out_channels)

        while out_channels % groups != 0:
            groups -= 1

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),
            nn.GroupNorm(
                groups,
                out_channels
            ),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Decoder block with encoder skip connections
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):
        super().__init__()

        self.fuse = nn.Sequential(
            ConvNormAct(
                in_channels + skip_channels,
                out_channels
            ),
            ConvNormAct(
                out_channels,
                out_channels
            )
        )

    def forward(
        self,
        x,
        skip
    ):

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat(
            [x, skip],
            dim=1
        )

        return self.fuse(x)


# ============================================================
# ResNet-18 based colorization model
# ============================================================

class ResNet18Colorizer(nn.Module):

    def __init__(
        self,
        pretrained=True
    ):
        super().__init__()

        # ImageNet-pretrained encoder.
        # The weights may download the first time this runs.
        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.encoder = resnet18(
            weights=weights
        )

        # Expected ImageNet normalization.
        # The grayscale L channel is replicated to three channels.
        self.register_buffer(
            "rgb_mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "rgb_std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1)
        )

        # ResNet-18 encoder feature sizes for a 256x256 input:
        #
        # Stem       : 64 channels, 128x128
        # Layer 1    : 64 channels,  64x64
        # Layer 2    : 128 channels, 32x32
        # Layer 3    : 256 channels, 16x16
        # Layer 4    : 512 channels,  8x8

        # Multi-scale decoder.
        self.decode3 = DecoderBlock(
            512, 256, 256
        )

        self.decode2 = DecoderBlock(
            256, 128, 128
        )

        self.decode1 = DecoderBlock(
            128, 64, 96
        )

        self.decode0 = DecoderBlock(
            96, 64, 64
        )

        # Final full-resolution color refinement.
        self.refine = nn.Sequential(
            ConvNormAct(
                65,
                32
            ),
            ConvNormAct(
                32,
                32
            ),
            nn.Conv2d(
                32,
                2,
                kernel_size=1
            ),
            nn.Tanh()
        )

    def forward(self, L):

        # L shape: [B, 1, H, W], normalized to [0, 1].
        if L.ndim != 4 or L.shape[1] != 1:
            raise ValueError(
                "Expected input with shape [B, 1, H, W]."
            )

        # Convert grayscale input into three channels.
        rgb = L.repeat(
            1, 3, 1, 1
        )

        rgb = (
            rgb - self.rgb_mean
        ) / self.rgb_std

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        e0 = self.encoder.relu(
            self.encoder.bn1(
                self.encoder.conv1(rgb)
            )
        )

        e1 = self.encoder.layer1(
            self.encoder.maxpool(e0)
        )

        e2 = self.encoder.layer2(e1)

        e3 = self.encoder.layer3(e2)

        e4 = self.encoder.layer4(e3)

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        d3 = self.decode3(
            e4, e3
        )

        d2 = self.decode2(
            d3, e2
        )

        d1 = self.decode1(
            d2, e1
        )

        d0 = self.decode0(
            d1, e0
        )

        # Restore the original input resolution.
        d0 = F.interpolate(
            d0,
            size=L.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Include the input lightness as an additional feature.
        combined = torch.cat(
            [d0, L],
            dim=1
        )

        # Predict normalized A/B color channels.
        output_AB = self.refine(
            combined
        )

        return output_AB


# ============================================================
# Architecture and GPU memory test
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("RESNET-18 COLORIZATION MODEL TEST")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("PyTorch:", torch.__version__)
    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print()
    print("Loading ImageNet-pretrained ResNet-18...")

    model = ResNet18Colorizer(
        pretrained=True
    ).to(device)

    model.eval()

    # Test on one batch without gradients.
    test_input = torch.rand(
        2, 1, 256, 256,
        device=device
    )

    print("Input shape:", test_input.shape)

    with torch.inference_mode():

        output = model(
            test_input
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Output shape:", output.shape)

    print(
        "Output range:",
        output.min().item(),
        "to",
        output.max().item()
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    if device.type == "cuda":

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        print(
            "Peak GPU memory:",
            f"{peak_memory:.2f} MB"
        )

    assert output.shape == (
        2, 2, 256, 256
    )

    assert torch.isfinite(output).all().item()

    print()
    print("ResNet-18 architecture test: PASSED")
    print("=" * 65)