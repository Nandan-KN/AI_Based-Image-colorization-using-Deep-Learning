import torch
import torch.nn as nn
import torch.nn.functional as F

from advanced_color_loss import AdvancedColorLoss


class TeacherGuidedColorLoss(nn.Module):

    def __init__(
        self,
        teacher_weight=0.2
    ):
        super().__init__()

        self.original_loss = AdvancedColorLoss(
            color_weight=1.0,
            chroma_weight=1.5,
            edge_weight=0.15
        )

        self.teacher_weight = teacher_weight

    def forward(
        self,
        predicted_AB,
        ground_truth_AB,
        teacher_AB
    ):
        # Preserve supervision from the original photograph.
        original_losses = self.original_loss(
            predicted_AB,
            ground_truth_AB
        )

        # Additional guidance from DDColor.
        teacher_loss = F.smooth_l1_loss(
            predicted_AB,
            teacher_AB
        )

        total_loss = (
            original_losses["total"]
            + self.teacher_weight * teacher_loss
        )

        return {
            "total": total_loss,
            "original": original_losses["total"],
            "color": original_losses["color"],
            "chroma": original_losses["chroma"],
            "edge": original_losses["edge"],
            "teacher": teacher_loss
        }


if __name__ == "__main__":

    print("=" * 60)
    print("TEACHER-GUIDED LOSS TEST")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    predicted_AB = torch.randn(
        2, 2, 256, 256,
        device=device
    ).clamp(-1.0, 1.0)

    ground_truth_AB = torch.randn_like(
        predicted_AB
    ).clamp(-1.0, 1.0)

    teacher_AB = torch.randn_like(
        predicted_AB
    ).clamp(-1.0, 1.0)

    predicted_AB.requires_grad_(True)

    criterion = TeacherGuidedColorLoss(
        teacher_weight=0.2
    ).to(device)

    losses = criterion(
        predicted_AB,
        ground_truth_AB,
        teacher_AB
    )

    for name, value in losses.items():
        print(f"{name.capitalize()} loss: {value.item():.6f}")

    losses["total"].backward()

    assert torch.isfinite(losses["total"]).item()
    assert predicted_AB.grad is not None
    assert torch.isfinite(predicted_AB.grad).all().item()

    print("Gradient finite:", True)
    print("Teacher-guided loss test: PASSED")
    print("=" * 60)