import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic convolution block
# ============================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.GroupNorm(
                num_groups=min(8, out_channels),
                num_channels=out_channels
            ),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Residual block
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        groups = min(8, channels)

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm1 = nn.GroupNorm(
            groups,
            channels
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm2 = nn.GroupNorm(
            groups,
            channels
        )

        self.act = nn.GELU()

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual
        x = self.act(x)

        return x


# ============================================================
# Multi-scale context
# ============================================================

class MultiScaleContext(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.branch1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=False
        )

        self.branch2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            bias=False
        )

        self.branch3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=4,
            dilation=4,
            bias=False
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                channels * 3,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.GroupNorm(
                min(8, channels),
                channels
            ),
            nn.GELU()
        )

    def forward(self, x):

        a = self.branch1(x)
        b = self.branch2(x)
        c = self.branch3(x)

        x = torch.cat(
            [a, b, c],
            dim=1
        )

        return self.project(x)


# ============================================================
# Downsampling encoder block
# ============================================================

class DownBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.GroupNorm(
                min(8, out_channels),
                out_channels
            ),

            nn.GELU(),

            ResidualBlock(
                out_channels
            )
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Pixel decoder up block
# ============================================================

class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.fusion = ConvBlock(
            out_channels + skip_channels,
            out_channels
        )

        self.residual = ResidualBlock(
            out_channels
        )

    def forward(
        self,
        x,
        skip
    ):

        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:

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

        x = self.fusion(x)

        x = self.residual(x)

        return x


# ============================================================
# Color query decoder
# ============================================================

class ColorQueryDecoder(nn.Module):

    def __init__(
        self,
        feature_channels=128,
        query_dim=64,
        num_queries=16,
        num_heads=4
    ):
        super().__init__()

        self.num_queries = num_queries
        self.query_dim = query_dim

        # Visual features -> attention dimension
        self.visual_projection = nn.Linear(
            feature_channels,
            query_dim
        )

        # Learnable color queries
        self.query_tokens = nn.Parameter(
            torch.randn(
                1,
                num_queries,
                query_dim
            ) * 0.02
        )

        self.query_norm = nn.LayerNorm(
            query_dim
        )

        # Query self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Queries attend to image features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Feed-forward refinement
        self.feed_forward = nn.Sequential(

            nn.LayerNorm(
                query_dim
            ),

            nn.Linear(
                query_dim,
                query_dim * 2
            ),

            nn.GELU(),

            nn.Linear(
                query_dim * 2,
                query_dim
            )
        )

        # Convert each learned query into an A/B color
        self.query_to_ab = nn.Sequential(

            nn.LayerNorm(
                query_dim
            ),

            nn.Linear(
                query_dim,
                32
            ),

            nn.GELU(),

            nn.Linear(
                32,
                2
            ),

            nn.Tanh()
        )

    def forward(self, bottleneck):

        batch_size, channels, height, width = bottleneck.shape

        # ----------------------------------------------------
        # Convert image feature map into visual tokens
        # ----------------------------------------------------

        visual_tokens = bottleneck.flatten(
            2
        ).transpose(
            1,
            2
        )

        visual_tokens = self.visual_projection(
            visual_tokens
        )

        # ----------------------------------------------------
        # Create query tokens
        # ----------------------------------------------------

        queries = self.query_tokens.expand(
            batch_size,
            -1,
            -1
        )

        # ----------------------------------------------------
        # Query self-attention
        # ----------------------------------------------------

        normalized_queries = self.query_norm(
            queries
        )

        self_attended, _ = self.self_attention(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            need_weights=False
        )

        queries = queries + self_attended

        # ----------------------------------------------------
        # Query -> image cross-attention
        # ----------------------------------------------------

        cross_attended, _ = self.cross_attention(
            queries,
            visual_tokens,
            visual_tokens,
            need_weights=False
        )

        queries = queries + cross_attended

        # ----------------------------------------------------
        # Feed-forward refinement
        # ----------------------------------------------------

        queries = queries + self.feed_forward(
            queries
        )

        # ----------------------------------------------------
        # Each query predicts a possible color
        # ----------------------------------------------------

        query_colors = self.query_to_ab(
            queries
        )

        return queries, query_colors


# ============================================================
# Main Advanced Model
# ============================================================

class AdvancedDDColorInspired(nn.Module):

    def __init__(
        self,
        num_queries=16
    ):
        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        # 256 x 256
        self.stem = nn.Sequential(

            ConvBlock(
                1,
                32
            ),

            ResidualBlock(
                32
            )
        )

        # 128 x 128
        self.down1 = DownBlock(
            32,
            48
        )

        # 64 x 64
        self.down2 = DownBlock(
            48,
            64
        )

        # 32 x 32
        self.down3 = DownBlock(
            64,
            96
        )

        # 16 x 16
        self.down4 = DownBlock(
            96,
            128
        )

        # ----------------------------------------------------
        # Multi-scale feature processing
        # ----------------------------------------------------

        self.context = MultiScaleContext(
            128
        )

        self.bottleneck = ResidualBlock(
            128
        )

        # ----------------------------------------------------
        # Color-query decoder
        # ----------------------------------------------------

        self.color_decoder = ColorQueryDecoder(
            feature_channels=128,
            query_dim=64,
            num_queries=num_queries,
            num_heads=4
        )

        # ----------------------------------------------------
        # Pixel decoder
        # ----------------------------------------------------

        self.up1 = UpBlock(
            128,
            96,
            96
        )

        self.up2 = UpBlock(
            96,
            64,
            64
        )

        self.up3 = UpBlock(
            64,
            48,
            48
        )

        self.up4 = UpBlock(
            48,
            32,
            32
        )

        # ----------------------------------------------------
        # Pixel-based A/B prediction
        # ----------------------------------------------------

        self.pixel_ab = nn.Sequential(

            ConvBlock(
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

        # ----------------------------------------------------
        # Convert high-resolution pixel features into
        # query-compatible features
        # ----------------------------------------------------

        self.pixel_query_projection = nn.Conv2d(
            32,
            64,
            kernel_size=1
        )

        # ----------------------------------------------------
        # Final fusion
        # ----------------------------------------------------

        self.final_fusion = nn.Sequential(

            nn.Conv2d(
                4,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.GroupNorm(
                8,
                32
            ),

            nn.GELU(),

            ResidualBlock(
                32
            ),

            nn.Conv2d(
                32,
                2,
                kernel_size=1
            ),

            nn.Tanh()
        )

    def forward(self, x):

        # ====================================================
        # Encoder
        # ====================================================

        skip1 = self.stem(
            x
        )

        skip2 = self.down1(
            skip1
        )

        skip3 = self.down2(
            skip2
        )

        skip4 = self.down3(
            skip3
        )

        bottleneck = self.down4(
            skip4
        )

        # ====================================================
        # Multi-scale context
        # ====================================================

        bottleneck = self.context(
            bottleneck
        )

        bottleneck = self.bottleneck(
            bottleneck
        )

        # ====================================================
        # Color query decoder
        # ====================================================

        query_features, query_colors = self.color_decoder(
            bottleneck
        )

        # ====================================================
        # Pixel decoder
        # ====================================================

        pixel_features = self.up1(
            bottleneck,
            skip4
        )

        pixel_features = self.up2(
            pixel_features,
            skip3
        )

        pixel_features = self.up3(
            pixel_features,
            skip2
        )

        pixel_features = self.up4(
            pixel_features,
            skip1
        )

        # ====================================================
        # Direct pixel color prediction
        # ====================================================

        pixel_ab = self.pixel_ab(
            pixel_features
        )

        # ====================================================
        # Use learned color queries at full resolution
        # ====================================================

        pixel_query_features = self.pixel_query_projection(
            pixel_features
        )

        batch_size, channels, height, width = (
            pixel_query_features.shape
        )

        pixel_tokens = pixel_query_features.flatten(
            2
        ).transpose(
            1,
            2
        )

        # Normalize pixel and query representations
        pixel_tokens = F.normalize(
            pixel_tokens,
            dim=-1
        )

        query_features = F.normalize(
            query_features,
            dim=-1
        )

        # ----------------------------------------------------
        # Similarity between every pixel and every color query
        # ----------------------------------------------------

        similarity = torch.matmul(
            pixel_tokens,
            query_features.transpose(
                1,
                2
            )
        )

        similarity = similarity / (
            self.color_decoder.query_dim ** 0.5
        )

        attention = torch.softmax(
            similarity,
            dim=-1
        )

        # ----------------------------------------------------
        # Weighted combination of query colors
        # ----------------------------------------------------

        query_ab = torch.matmul(
            attention,
            query_colors
        )

        query_ab = query_ab.transpose(
            1,
            2
        ).reshape(
            batch_size,
            2,
            height,
            width
        )

        # ====================================================
        # Final spatial + semantic color fusion
        # ====================================================

        combined = torch.cat(
            [
                pixel_ab,
                query_ab
            ],
            dim=1
        )

        output = self.final_fusion(
            combined
        )

        return output


# ============================================================
# Architecture test
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 60)
    print("ADVANCED DDColor-INSPIRED MODEL TEST")
    print("=" * 60)

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

    model = AdvancedDDColorInspired(
        num_queries=16
    ).to(device)

    # One test batch
    test_input = torch.randn(
        2,
        1,
        256,
        256,
        device=device
    )

    print(
        "Input shape:",
        test_input.shape
    )

    with torch.no_grad():

        test_output = model(
            test_input
        )

    print(
        "Output shape:",
        test_output.shape
    )

    print(
        "Output range:",
        test_output.min().item(),
        "to",
        test_output.max().item()
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    if torch.cuda.is_available():

        memory_allocated = (
            torch.cuda.memory_allocated()
            / (1024 ** 2)
        )

        memory_reserved = (
            torch.cuda.memory_reserved()
            / (1024 ** 2)
        )

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        print(
            "GPU memory allocated:",
            f"{memory_allocated:.2f} MB"
        )

        print(
            "GPU memory reserved:",
            f"{memory_reserved:.2f} MB"
        )

        print(
            "Peak GPU memory:",
            f"{peak_memory:.2f} MB"
        )

    assert test_output.shape == (
        2,
        2,
        256,
        256
    )

    print()
    print(
        "Architecture test: PASSED"
    )

    print("=" * 60)