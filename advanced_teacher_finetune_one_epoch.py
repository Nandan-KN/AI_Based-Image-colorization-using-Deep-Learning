import os
import time

import torch
from torch.utils.data import DataLoader

from advanced_ddcolor_model import AdvancedDDColorInspired
from advanced_teacher_loader import AdvancedTeacherDataset
from advanced_split_loader import AdvancedSplitDataset
from advanced_color_loss import AdvancedColorLoss
from advanced_teacher_loss import TeacherGuidedColorLoss


BASE_CHECKPOINT = "models/advanced_ddcolor_best.pth"
OUTPUT_CHECKPOINT = "models/advanced_ddcolor_teacher_epoch1.pth"

BATCH_SIZE = 2
LEARNING_RATE = 1e-5
TEACHER_WEIGHT = 5.0
IMAGE_SIZE = 256


def evaluate_validation(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    count = 0

    with torch.inference_mode():
        for L, target_AB in loader:
            L = L.to(device)
            target_AB = target_AB.to(device)

            prediction = model(L)
            losses = criterion(prediction, target_AB)

            if not torch.isfinite(losses["total"]).item():
                raise RuntimeError("Non-finite validation loss.")

            total_loss += losses["total"].item()
            count += 1

    if count == 0:
        raise RuntimeError("Validation loader is empty.")

    return total_loss / count


def main():
    print("=" * 65)
    print("ONE-EPOCH TEACHER-GUIDED FINE-TUNING")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load the original best model, not the pilot.
    checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location=device
    )

    model = AdvancedDDColorInspired(
        num_queries=checkpoint.get("num_queries", 16)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Starting checkpoint:", BASE_CHECKPOINT)
    print("Starting model epoch:", checkpoint["epoch"])

    # Training data: original image + DDColor teacher output.
    train_dataset = AdvancedTeacherDataset(
        manifest="advanced_dataset/splits/train.txt",
        image_root="advanced_dataset/images",
        teacher_root="advanced_dataset/teacher_outputs/train",
        image_size=IMAGE_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Validation uses original photographs only.
    validation_dataset = AdvancedSplitDataset(
        split_file="advanced_dataset/splits/validation.txt",
        image_root="advanced_dataset/images",
        image_size=IMAGE_SIZE
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Training batches:", len(train_loader))

    # Always evaluate using the original ground-truth objective.
    original_criterion = AdvancedColorLoss().to(device)

    # Training objective combines ground truth and teacher guidance.
    teacher_criterion = TeacherGuidedColorLoss(
        teacher_weight=TEACHER_WEIGHT
    ).to(device)

    print("\nEvaluating baseline validation loss...")
    validation_before = evaluate_validation(
        model,
        validation_loader,
        original_criterion,
        device
    )

    print(f"Validation before: {validation_before:.6f}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    model.train()

    total_loss_sum = 0.0
    batches_completed = 0
    start_time = time.time()

    for batch_index, (L, target_AB, teacher_AB) in enumerate(
        train_loader,
        start=1
    ):
        L = L.to(device)
        target_AB = target_AB.to(device)
        teacher_AB = teacher_AB.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Full precision for this experiment.
        prediction = model(L)

        losses = teacher_criterion(
            prediction,
            target_AB,
            teacher_AB
        )

        if not torch.isfinite(losses["total"]).item():
            raise RuntimeError(
                f"Non-finite loss at batch {batch_index}."
            )

        losses["total"].backward()

        # Stop immediately if any gradient becomes non-finite.
        for parameter in model.parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(
                        f"Non-finite gradient at batch {batch_index}."
                    )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss_sum += losses["total"].item()
        batches_completed += 1

        if batch_index % 500 == 0 or batch_index == 1:
            print(
                f"Batch {batch_index}/{len(train_loader)} | "
                f"Total: {losses['total'].item():.6f} | "
                f"Original: {losses['original'].item():.6f} | "
                f"Teacher: {losses['teacher'].item():.6f}"
            )

    average_train_loss = total_loss_sum / batches_completed

    print("\nEvaluating validation loss after fine-tuning...")
    validation_after = evaluate_validation(
        model,
        validation_loader,
        original_criterion,
        device
    )

    os.makedirs("models", exist_ok=True)

    torch.save(
        {
            "epoch": checkpoint["epoch"] + 1,
            "finetune_epoch": 1,
            "base_checkpoint_epoch": checkpoint["epoch"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_queries": checkpoint.get("num_queries", 16),
            "image_size": IMAGE_SIZE,
            "teacher_weight": TEACHER_WEIGHT,
            "average_train_loss": average_train_loss,
            "validation_before": validation_before,
            "validation_after": validation_after
        },
        OUTPUT_CHECKPOINT
    )

    print("\n" + "=" * 65)
    print("ONE-EPOCH FINE-TUNING COMPLETED")
    print("=" * 65)
    print("Batches completed:", batches_completed)
    print(f"Average training loss: {average_train_loss:.6f}")
    print(f"Validation before: {validation_before:.6f}")
    print(f"Validation after : {validation_after:.6f}")
    print("New checkpoint:", OUTPUT_CHECKPOINT)
    print(f"Elapsed time: {(time.time() - start_time) / 60:.1f} minutes")

    if validation_after < validation_before:
        print("Validation loss improved.")
    else:
        print("Validation loss did not improve.")

    print("Original best checkpoint remains unchanged.")
    print("=" * 65)


if __name__ == "__main__":
    main()