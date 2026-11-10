import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedColorLoss(nn.Module):

    def __init__(
        self,
        color_weight=1.0,
        chroma_weight=1.5,
        edge_weight=0.15
    ):
        super().__init__()

        self.color_weight = color_weight
        self.chroma_weight = chroma_weight
        self.edge_weight = edge_weight

    # --------------------------------------------------------
    # Color reconstruction loss
    # --------------------------------------------------------

    def color_loss(
        self,
        predicted,
        target
    ):

        return F.smooth_l1_loss(
            predicted,
            target
        )

    # --------------------------------------------------------
    # Chroma-aware loss
    # --------------------------------------------------------

    def chroma_loss(
        self,
        predicted,
        target
    ):

        predicted_chroma = torch.sqrt(
            predicted[:, 0:1] ** 2
            +
            predicted[:, 1:2] ** 2
            +
            1e-8
        )

        target_chroma = torch.sqrt(
            target[:, 0:1] ** 2
            +
            target[:, 1:2] ** 2
            +
            1e-8
        )

        # Give more importance to strongly colored regions.
        strength = (
            target_chroma
            / (
                target_chroma.amax(
                    dim=(2, 3),
                    keepdim=True
                )
                + 1e-8
            )
        )

        weighted_difference = (
            torch.abs(
                predicted - target
            )
            * (
                1.0
                + self.chroma_weight * strength
            )
        )

        return weighted_difference.mean()

    # --------------------------------------------------------
    # Spatial gradient
    # --------------------------------------------------------

    @staticmethod
    def gradient_x(x):

        return x[:, :, :, 1:] - x[:, :, :, :-1]

    @staticmethod
    def gradient_y(x):

        return x[:, :, 1:, :] - x[:, :, :-1, :]

    # --------------------------------------------------------
    # Edge consistency loss
    # --------------------------------------------------------

    def edge_loss(
        self,
        predicted,
        target
    ):

        predicted_x = self.gradient_x(
            predicted
        )

        target_x = self.gradient_x(
            target
        )

        predicted_y = self.gradient_y(
            predicted
        )

        target_y = self.gradient_y(
            target
        )

        loss_x = F.l1_loss(
            predicted_x,
            target_x
        )

        loss_y = F.l1_loss(
            predicted_y,
            target_y
        )

        return (
            loss_x
            +
            loss_y
        ) / 2.0

    # --------------------------------------------------------
    # Final loss
    # --------------------------------------------------------

    def forward(
        self,
        predicted,
        target
    ):

        color = self.color_loss(
            predicted,
            target
        )

        chroma = self.chroma_loss(
            predicted,
            target
        )

        edge = self.edge_loss(
            predicted,
            target
        )

        total = (
            self.color_weight * color
            +
            self.chroma_weight * chroma
            +
            self.edge_weight * edge
        )

        return {
            "total": total,
            "color": color,
            "chroma": chroma,
            "edge": edge
        }


# ============================================================
# One-batch loss test
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("ADVANCED COLOR LOSS TEST")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    predicted = torch.randn(
        2,
        2,
        256,
        256,
        device=device
    ).clamp(
        -1.0,
        1.0
    )

    target = torch.randn(
        2,
        2,
        256,
        256,
        device=device
    ).clamp(
        -1.0,
        1.0
    )

    predicted.requires_grad_(True)

    criterion = AdvancedColorLoss(
        color_weight=1.0,
        chroma_weight=1.5,
        edge_weight=0.15
    )

    losses = criterion(
        predicted,
        target
    )

    print(
        "Color loss:",
        losses["color"].item()
    )

    print(
        "Chroma loss:",
        losses["chroma"].item()
    )

    print(
        "Edge loss:",
        losses["edge"].item()
    )

    print(
        "Total loss:",
        losses["total"].item()
    )

    losses["total"].backward()

    print(
        "Gradient finite:",
        torch.isfinite(
            predicted.grad
        ).all().item()
    )

    assert torch.isfinite(
        losses["total"]
    ).item()

    assert torch.isfinite(
        predicted.grad
    ).all().item()

    print()
    print(
        "Advanced color loss test: PASSED"
    )

    print("=" * 60)