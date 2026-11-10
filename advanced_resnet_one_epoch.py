import os
import time

import torch
from torch.utils.data import DataLoader

from advanced_resnet_colorizer import ResNet18Colorizer
from advanced_split_loader import AdvancedSplitDataset
from advanced_color_loss import AdvancedColorLoss


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 2
IMAGE_SIZE = 256

PILOT_BATCHES = 9000

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

OUTPUT_CHECKPOINT = "models/resnet18_colorizer_epoch1.pth"


# ============================================================
# Validation
# ============================================================

def evaluate_validation(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    total_loss = 0.0
    batches = 0

    with torch.inference_mode():

        for L, target_AB in loader:

            L = L.to(
                device,
                non_blocking=True
            )

            target_AB = target_AB.to(
                device,
                non_blocking=True
            )

            prediction = model(L)

            losses = criterion(
                prediction,
                target_AB
            )

            if not torch.isfinite(
                losses["total"]
            ).item():

                raise RuntimeError(
                    "Non-finite validation loss."
                )

            total_loss += losses["total"].item()
            batches += 1

    if batches == 0:
        raise RuntimeError("Validation loader is empty.")

    return total_loss / batches


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 65)
    print("RESNET-18 COLORIZATION PILOT")
    print("=" * 65)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    train_dataset = AdvancedSplitDataset(
        split_file="advanced_dataset/splits/train.txt",
        image_root="advanced_dataset/images",
        image_size=IMAGE_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Initialize pretrained ResNet-18 model
    # --------------------------------------------------------

    model = ResNet18Colorizer(
        pretrained=True
    ).to(device)

    # Freeze the pretrained encoder for this pilot.
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in trainable_parameters):,}"
    )

    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    criterion = AdvancedColorLoss().to(device)

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Baseline validation
    # --------------------------------------------------------

    print("\nEvaluating before training...")

    validation_before = evaluate_validation(
        model,
        validation_loader,
        criterion,
        device
    )

    print(
        f"Validation loss before: "
        f"{validation_before:.6f}"
    )

    # --------------------------------------------------------
    # Train the decoder for a limited number of batches
    # --------------------------------------------------------

    model.train()

    # Keep the frozen encoder in evaluation mode so its
    # pretrained BatchNorm statistics remain unchanged.
    model.encoder.eval()

    total_train_loss = 0.0
    batches_completed = 0

    start_time = time.time()

    for batch_index, (L, target_AB) in enumerate(
        train_loader,
        start=1
    ):

        L = L.to(
            device,
            non_blocking=True
        )

        target_AB = target_AB.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(L)

        losses = criterion(
            prediction,
            target_AB
        )

        total_loss = losses["total"]

        if not torch.isfinite(
            total_loss
        ).item():

            raise RuntimeError(
                f"Non-finite training loss at batch "
                f"{batch_index}."
            )

        total_loss.backward()

        # Check for invalid gradients.
        for parameter in trainable_parameters:

            if parameter.grad is not None:

                if not torch.isfinite(
                    parameter.grad
                ).all():

                    raise RuntimeError(
                        f"Invalid gradient at batch "
                        f"{batch_index}."
                    )

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0
        )

        optimizer.step()

        total_train_loss += total_loss.item()
        batches_completed += 1

        if (
            batch_index == 1
            or batch_index % 50 == 0
        ):

            print(
                f"Batch {batch_index}/{PILOT_BATCHES} | "
                f"Loss: {total_loss.item():.6f}"
            )

        if batch_index >= PILOT_BATCHES:
            break

    if batches_completed == 0:
        raise RuntimeError("No training batches completed.")

    average_train_loss = (
        total_train_loss / batches_completed
    )

    # --------------------------------------------------------
    # Validation after training
    # --------------------------------------------------------

    print("\nEvaluating after training...")

    validation_after = evaluate_validation(
        model,
        validation_loader,
        criterion,
        device
    )

    # --------------------------------------------------------
    # Save separate experimental checkpoint
    # --------------------------------------------------------

    os.makedirs("models", exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": IMAGE_SIZE,
            "pretrained_encoder": True,
            "encoder_frozen": True,
            "pilot_batches": batches_completed,
            "validation_before": validation_before,
            "validation_after": validation_after,
            "average_train_loss": average_train_loss
        },
        OUTPUT_CHECKPOINT
    )

    elapsed_minutes = (
        time.time() - start_time
    ) / 60.0

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("RESNET-18 PILOT COMPLETED")
    print("=" * 65)

    print("Batches completed:", batches_completed)

    print(
        f"Average training loss: "
        f"{average_train_loss:.6f}"
    )

    print(
        f"Validation before: "
        f"{validation_before:.6f}"
    )

    print(
        f"Validation after : "
        f"{validation_after:.6f}"
    )

    print(
        "New checkpoint:",
        OUTPUT_CHECKPOINT
    )

    print(
        f"Elapsed time: {elapsed_minutes:.1f} minutes"
    )

    if validation_after < validation_before:

        print(
            "Validation loss improved in this pilot."
        )

    else:

        print(
            "Validation loss did not improve in this pilot."
        )

    print(
        "Existing advanced model checkpoints remain unchanged."
    )

    print("=" * 65)


if __name__ == "__main__":
    main()