import os

import torch
from torch.utils.data import DataLoader

from advanced_resnet_colorizer import ResNet18Colorizer
from advanced_split_loader import AdvancedSplitDataset
from advanced_color_loss import AdvancedColorLoss
from resnet_chroma_loss import ResNetChromaAwareLoss


# ============================================================
# Configuration
# ============================================================

BASE_CHECKPOINT = "models/resnet18_colorizer_finetuned.pth"
OUTPUT_CHECKPOINT = "models/resnet18_chroma_pilot.pth"

BATCH_SIZE = 2
IMAGE_SIZE = 256
PILOT_BATCHES = 500

ENCODER_LR = 1e-5
DECODER_LR = 1e-4
WEIGHT_DECAY = 1e-5

CHROMA_MAGNITUDE_WEIGHT = 0.5


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, loader, base_criterion, chroma_criterion, device):
    model.eval()

    base_loss_sum = 0.0
    magnitude_loss_sum = 0.0
    batches = 0

    with torch.inference_mode():
        for L, target_AB in loader:
            L = L.to(device, non_blocking=True)
            target_AB = target_AB.to(device, non_blocking=True)

            prediction = model(L)

            base_losses = base_criterion(
                prediction,
                target_AB
            )

            chroma_losses = chroma_criterion(
                prediction,
                target_AB
            )

            if not torch.isfinite(base_losses["total"]).item():
                raise RuntimeError(
                    "Non-finite validation loss detected."
                )

            if not torch.isfinite(
                chroma_losses["chroma_magnitude"]
            ).item():
                raise RuntimeError(
                    "Non-finite chroma-magnitude loss detected."
                )

            base_loss_sum += base_losses["total"].item()
            magnitude_loss_sum += (
                chroma_losses["chroma_magnitude"].item()
            )

            batches += 1

    if batches == 0:
        raise RuntimeError("Validation loader is empty.")

    return (
        base_loss_sum / batches,
        magnitude_loss_sum / batches
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print("RESNET-18 CHROMA-AWARE PILOT")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()

    if not os.path.isfile(BASE_CHECKPOINT):
        raise FileNotFoundError(BASE_CHECKPOINT)

    # --------------------------------------------------------
    # Load the fine-tuned ResNet-18 checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location=device
    )

    model = ResNet18Colorizer(
        pretrained=False
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Starting checkpoint:", BASE_CHECKPOINT)

    # Freeze all model parameters first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze the final ResNet encoder layer.
    for parameter in model.encoder.layer4.parameters():
        parameter.requires_grad = True

    # Unfreeze the colorization decoder.
    for name, parameter in model.named_parameters():
        if not name.startswith("encoder."):
            parameter.requires_grad = True

    encoder_parameters = list(
        model.encoder.layer4.parameters()
    )

    decoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]

    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = AdvancedSplitDataset(
        split_file="advanced_dataset/splits/train.txt",
        image_root="advanced_dataset/images",
        image_size=IMAGE_SIZE
    )

    validation_dataset = AdvancedSplitDataset(
        split_file="advanced_dataset/splits/validation.txt",
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

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Training batches available:", len(train_loader))

    # --------------------------------------------------------
    # Losses
    # --------------------------------------------------------

    base_criterion = AdvancedColorLoss().to(device)

    chroma_criterion = ResNetChromaAwareLoss(
        chroma_magnitude_weight=CHROMA_MAGNITUDE_WEIGHT
    ).to(device)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": ENCODER_LR
            },
            {
                "params": decoder_parameters,
                "lr": DECODER_LR
            }
        ],
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Validation before training
    # --------------------------------------------------------

    print("\nEvaluating before chroma-aware training...")

    validation_before, magnitude_before = evaluate_model(
        model,
        validation_loader,
        base_criterion,
        chroma_criterion,
        device
    )

    print(f"Base validation loss before: {validation_before:.6f}")
    print(f"Chroma magnitude loss before: {magnitude_before:.6f}")

    # --------------------------------------------------------
    # Train for a limited number of batches
    # --------------------------------------------------------

    model.train()

    # Preserve pretrained BatchNorm statistics in the encoder.
    model.encoder.eval()

    training_loss_sum = 0.0
    batches_completed = 0

    for batch_index, (L, target_AB) in enumerate(
        train_loader,
        start=1
    ):
        L = L.to(device, non_blocking=True)
        target_AB = target_AB.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        prediction = model(L)

        losses = chroma_criterion(
            prediction,
            target_AB
        )

        total_loss = losses["total"]

        if not torch.isfinite(total_loss).item():
            raise RuntimeError(
                f"Non-finite loss at batch {batch_index}."
            )

        total_loss.backward()

        for parameter in model.parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(
                        f"Non-finite gradient at batch {batch_index}."
                    )

        torch.nn.utils.clip_grad_norm_(
            [
                p for p in model.parameters()
                if p.requires_grad
            ],
            max_norm=1.0
        )

        optimizer.step()

        training_loss_sum += total_loss.item()
        batches_completed += 1

        if batch_index == 1 or batch_index % 50 == 0:
            print(
                f"Batch {batch_index}/{PILOT_BATCHES} | "
                f"Total: {total_loss.item():.6f} | "
                f"Color: {losses['color'].item():.6f} | "
                f"Chroma magnitude: "
                f"{losses['chroma_magnitude'].item():.6f}"
            )

        if batch_index >= PILOT_BATCHES:
            break

    if batches_completed == 0:
        raise RuntimeError("No training batches completed.")

    # --------------------------------------------------------
    # Validation after training
    # --------------------------------------------------------

    print("\nEvaluating after chroma-aware training...")

    validation_after, magnitude_after = evaluate_model(
        model,
        validation_loader,
        base_criterion,
        chroma_criterion,
        device
    )

    average_training_loss = (
        training_loss_sum / batches_completed
    )

    # --------------------------------------------------------
    # Save separate checkpoint
    # --------------------------------------------------------

    os.makedirs("models", exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "image_size": IMAGE_SIZE,
            "base_checkpoint": BASE_CHECKPOINT,
            "pilot_batches": batches_completed,
            "chroma_magnitude_weight": CHROMA_MAGNITUDE_WEIGHT,
            "average_training_loss": average_training_loss,
            "validation_before": validation_before,
            "validation_after": validation_after,
            "magnitude_before": magnitude_before,
            "magnitude_after": magnitude_after
        },
        OUTPUT_CHECKPOINT
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("CHROMA-AWARE PILOT COMPLETED")
    print("=" * 65)

    print("Batches completed:", batches_completed)
    print(f"Average training loss: {average_training_loss:.6f}")
    print(f"Base validation loss before: {validation_before:.6f}")
    print(f"Base validation loss after : {validation_after:.6f}")
    print(f"Chroma magnitude before: {magnitude_before:.6f}")
    print(f"Chroma magnitude after : {magnitude_after:.6f}")
    print("New checkpoint:", OUTPUT_CHECKPOINT)

    if validation_after < validation_before:
        print("Base validation loss improved.")
    else:
        print("Base validation loss did not improve.")

    print("Previous checkpoints remain unchanged.")
    print("=" * 65)


if __name__ == "__main__":
    main()