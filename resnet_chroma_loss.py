import torch
import torch.nn as nn
import torch.nn.functional as F

from advanced_color_loss import AdvancedColorLoss


class ResNetChromaAwareLoss(nn.Module):

    def __init__(
        self,
        chroma_magnitude_weight=0.5
    ):
        super().__init__()

        self.base_loss = AdvancedColorLoss(
            color_weight=1.0,
            chroma_weight=1.5,
            edge_weight=0.15
        )

        self.chroma_magnitude_weight = (
            chroma_magnitude_weight
        )

    def forward(
        self,
        predicted_AB,
        target_AB
    ):
        # Existing color, chroma-weighted and edge losses.
        base = self.base_loss(
            predicted_AB,
            target_AB
        )

        # Calculate chroma magnitude at each pixel.
        predicted_chroma = torch.sqrt(
            predicted_AB[:, 0:1] ** 2
            + predicted_AB[:, 1:2] ** 2
            + 1e-8
        )

        target_chroma = torch.sqrt(
            target_AB[:, 0:1] ** 2
            + target_AB[:, 1:2] ** 2
            + 1e-8
        )

        # Explicitly match predicted color strength to the target.
        magnitude_loss = F.smooth_l1_loss(
            predicted_chroma,
            target_chroma
        )

        total = (
            base["total"]
            + self.chroma_magnitude_weight * magnitude_loss
        )

        return {
            "total": total,
            "color": base["color"],
            "chroma_weighted": base["chroma"],
            "edge": base["edge"],
            "chroma_magnitude": magnitude_loss
        }


if __name__ == "__main__":

    print("=" * 60)
    print("RESNET CHROMA-AWARE LOSS TEST")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    predicted_AB = torch.randn(
        2, 2, 256, 256,
        device=device
    ).clamp(-1, 1)

    target_AB = torch.randn_like(
        predicted_AB
    ).clamp(-1, 1)

    predicted_AB.requires_grad_(True)

    criterion = ResNetChromaAwareLoss(
        chroma_magnitude_weight=0.5
    ).to(device)

    losses = criterion(
        predicted_AB,
        target_AB
    )

    for name, value in losses.items():
        print(f"{name}: {value.item():.6f}")

    losses["total"].backward()

    assert torch.isfinite(losses["total"]).item()
    assert predicted_AB.grad is not None
    assert torch.isfinite(predicted_AB.grad).all().item()

    print("Gradient finite:", True)
    print("Chroma-aware loss test: PASSED")
    print("=" * 60)