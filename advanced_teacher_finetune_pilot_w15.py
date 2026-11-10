import os

import torch
from torch.utils.data import DataLoader

from advanced_ddcolor_model import AdvancedDDColorInspired
from advanced_teacher_loader import AdvancedTeacherDataset
from advanced_split_loader import AdvancedSplitDataset
from advanced_color_loss import AdvancedColorLoss
from advanced_teacher_loss import TeacherGuidedColorLoss


BASE_CHECKPOINT = "models/advanced_ddcolor_best.pth"
PILOT_CHECKPOINT = "models/advanced_teacher_pilot_w15.pth"

BATCH_SIZE = 2
PILOT_BATCHES = 200
LEARNING_RATE = 1e-5
TEACHER_WEIGHT = 15.0


def evaluate_validation(model, loader, criterion, device):
    """Evaluate only against original ground-truth colors."""

    model.eval()
    total_loss = 0.0
    batches = 0

    with torch.inference_mode():
        for L, target_AB in loader:
            L = L.to(device, non_blocking=True)
            target_AB = target_AB.to(device, non_blocking=True)

            prediction = model(L)
            losses = criterion(prediction, target_AB)

            if not torch.isfinite(losses["total"]).item():
                raise RuntimeError("Non-finite validation loss detected.")

            total_loss += losses["total"].item()
            batches += 1

    if batches == 0:
        raise RuntimeError("Validation loader is empty.")

    return total_loss / batches


def main():
    print("=" * 65)
    print("DDCOLOR-GUIDED FINE-TUNING PILOT")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()

    # Load original best model.
    checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location=device
    )

    model = AdvancedDDColorInspired(
        num_queries=checkpoint.get("num_queries", 16)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Starting checkpoint epoch:", checkpoint["epoch"])
    print("Starting best validation loss:",
          checkpoint["best_validation_loss"])

    # Teacher-guided training data.
    train_dataset = AdvancedTeacherDataset(
        manifest="advanced_dataset/splits/train.txt",
        image_root="advanced_dataset/images",
        teacher_root="advanced_dataset/teacher_outputs/train",
        image_size=256
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Validation uses ORIGINAL images and ground-truth colors only.
    val_dataset = AdvancedSplitDataset(
        split_file="advanced_dataset/splits/validation.txt",
        image_root="advanced_dataset/images",
        image_size=256
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    original_criterion = AdvancedColorLoss().to(device)

    teacher_criterion = TeacherGuidedColorLoss(
        teacher_weight=TEACHER_WEIGHT
    ).to(device)

    # Measure baseline before updates.
    print("\nEvaluating baseline on validation images...")
    validation_before = evaluate_validation(
        model, val_loader, original_criterion, device
    )

    print(f"Validation loss before pilot: {validation_before:.6f}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    # Run a short pilot only.
    model.train()
    steps = 0

    for L, ground_truth_AB, teacher_AB in train_loader:
        L = L.to(device, non_blocking=True)
        ground_truth_AB = ground_truth_AB.to(
            device, non_blocking=True
        )
        teacher_AB = teacher_AB.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        prediction = model(L)

        losses = teacher_criterion(
            prediction,
            ground_truth_AB,
            teacher_AB
        )

        if not torch.isfinite(losses["total"]).item():
            raise RuntimeError("Non-finite training loss detected.")

        losses["total"].backward()

        # Verify all present gradients before the update.
        for parameter in model.parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(
                        "Non-finite gradient detected; update stopped."
                    )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()
        steps += 1

        if steps % 25 == 0 or steps == 1:
            print(
                f"Pilot batch {steps}/{PILOT_BATCHES} | "
                f"Total: {losses['total'].item():.6f} | "
                f"Original: {losses['original'].item():.6f} | "
                f"Teacher: {losses['teacher'].item():.6f}"
            )

        if steps >= PILOT_BATCHES:
            break

    # Measure validation after updates, using the same ground-truth loss.
    print("\nEvaluating pilot model on validation images...")
    validation_after = evaluate_validation(
        model, val_loader, original_criterion, device
    )

    os.makedirs("models", exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_queries": checkpoint.get("num_queries", 16),
            "image_size": 256,
            "base_checkpoint": BASE_CHECKPOINT,
            "base_checkpoint_epoch": checkpoint["epoch"],
            "pilot_batches": steps,
            "teacher_weight": TEACHER_WEIGHT,
            "validation_before": validation_before,
            "validation_after": validation_after
        },
        PILOT_CHECKPOINT
    )

    print("\n" + "=" * 65)
    print("TEACHER-GUIDED PILOT COMPLETED")
    print("=" * 65)
    print("Batches completed:", steps)
    print(f"Validation before: {validation_before:.6f}")
    print(f"Validation after : {validation_after:.6f}")
    print("Pilot checkpoint:", PILOT_CHECKPOINT)

    if validation_after < validation_before:
        print("Validation loss improved in this pilot.")
    else:
        print("Validation loss did not improve in this pilot.")

    print("Original best checkpoint remains unchanged.")
    print("=" * 65)


if __name__ == "__main__":
    main()