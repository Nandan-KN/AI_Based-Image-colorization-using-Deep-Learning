import os
import time

import torch
from torch.utils.data import DataLoader

from advanced_resnet_colorizer import ResNet18Colorizer
from advanced_split_loader import AdvancedSplitDataset
from advanced_color_loss import AdvancedColorLoss


BASE_CHECKPOINT = "models/resnet18_colorizer_epoch1.pth"
OUTPUT_CHECKPOINT = "models/resnet18_colorizer_finetuned.pth"

BATCH_SIZE = 2
IMAGE_SIZE = 256

ENCODER_LR = 1e-5
DECODER_LR = 1e-4

WEIGHT_DECAY = 1e-5


def evaluate(model, loader, criterion, device):
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
                raise RuntimeError("Non-finite validation loss.")

            total_loss += losses["total"].item()
            batches += 1

    if batches == 0:
        raise RuntimeError("Validation loader is empty.")

    return total_loss / batches


def main():
    print("=" * 65)
    print("RESNET-18 FINAL-LAYER FINE-TUNING")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Load the existing ResNet-18 checkpoint
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

    print("Loaded checkpoint:", BASE_CHECKPOINT)

    # --------------------------------------------------------
    # Freeze the entire model first
    # --------------------------------------------------------
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze the final ResNet encoder layer.
    for parameter in model.encoder.layer4.parameters():
        parameter.requires_grad = True

    # Train the colorization decoder and output head.
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
    # Training and validation datasets
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
    print("Training batches:", len(train_loader))

    criterion = AdvancedColorLoss().to(device)

    # Separate learning rates for the encoder and decoder.
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
    # Evaluate before fine-tuning
    # --------------------------------------------------------
    print("\nEvaluating before fine-tuning...")

    validation_before = evaluate(
        model,
        validation_loader,
        criterion,
        device
    )

    print(f"Validation before: {validation_before:.6f}")

    # --------------------------------------------------------
    # One complete training epoch
    # --------------------------------------------------------
    start_time = time.time()
    running_loss = 0.0
    batches_completed = 0

    model.train()

    # Keep BatchNorm statistics fixed in the pretrained encoder.
    # Gradients still update layer4's trainable weights.
    model.encoder.eval()

    for batch_index, (L, target_AB) in enumerate(
        train_loader,
        start=1
    ):
        L = L.to(device, non_blocking=True)
        target_AB = target_AB.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        prediction = model(L)

        losses = criterion(
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
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            max_norm=1.0
        )

        optimizer.step()

        running_loss += total_loss.item()
        batches_completed += 1

        if batch_index % 500 == 0 or batch_index == 1:
            print(
                f"Batch {batch_index}/{len(train_loader)} | "
                f"Loss: {total_loss.item():.6f}"
            )

    average_train_loss = (
        running_loss / batches_completed
    )

    # --------------------------------------------------------
    # Evaluate after fine-tuning
    # --------------------------------------------------------
    print("\nEvaluating after fine-tuning...")

    validation_after = evaluate(
        model,
        validation_loader,
        criterion,
        device
    )

    # Save a separate checkpoint.
    os.makedirs("models", exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "image_size": IMAGE_SIZE,
            "pretrained_encoder": True,
            "fine_tuned_encoder_layer": "layer4",
            "base_checkpoint": BASE_CHECKPOINT,
            "train_batches": batches_completed,
            "average_train_loss": average_train_loss,
            "validation_before": validation_before,
            "validation_after": validation_after
        },
        OUTPUT_CHECKPOINT
    )

    print()
    print("=" * 65)
    print("RESNET-18 FINE-TUNING COMPLETED")
    print("=" * 65)

    print("Batches completed:", batches_completed)
    print(f"Average training loss: {average_train_loss:.6f}")
    print(f"Validation before: {validation_before:.6f}")
    print(f"Validation after:  {validation_after:.6f}")
    print("Checkpoint:", OUTPUT_CHECKPOINT)
    print(f"Elapsed time: {(time.time() - start_time) / 60:.1f} minutes")

    if validation_after < validation_before:
        print("Validation loss improved.")
    else:
        print("Validation loss did not improve.")

    print("Previous checkpoints remain unchanged.")
    print("=" * 65)


if __name__ == "__main__":
    main()