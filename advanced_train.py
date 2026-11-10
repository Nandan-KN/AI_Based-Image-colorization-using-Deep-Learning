import argparse
import os
import time

import torch
from torch.cuda.amp import autocast, GradScaler

from advanced_ddcolor_model import AdvancedDDColorInspired
from advanced_split_loader import create_advanced_loaders
from advanced_color_loss import AdvancedColorLoss


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 2
IMAGE_SIZE = 256

EPOCHS = 30

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

NUM_QUERIES = 16

MODEL_DIR = "models"
MODEL_PATH = "models/advanced_ddcolor_best.pth"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# Helpers
# ============================================================

def load_checkpoint(model, optimizer, scheduler, path):
    if not path or not os.path.exists(path):
        return 0, float("inf")

    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_validation_loss", float("inf"))

    print()
    print("Resuming from checkpoint:")
    print("  Path:", path)
    print("  Epoch:", checkpoint["epoch"])
    print("  Best validation loss:", best_val_loss)

    return start_epoch, best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("ADVANCED DDColor-INSPIRED MODEL TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader
    ) = create_advanced_loaders(
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    print()
    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Test images:", len(test_dataset))

    model = AdvancedDDColorInspired(num_queries=NUM_QUERIES).to(device)

    total_parameters = sum(p.numel() for p in model.parameters())
    print()
    print("Model parameters:", f"{total_parameters:,}")

    criterion = AdvancedColorLoss(
        color_weight=1.0,
        chroma_weight=1.5,
        edge_weight=0.15
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    start_epoch, best_validation_loss = load_checkpoint(
        model,
        optimizer,
        scheduler,
        args.resume if args.resume else MODEL_PATH
    )

    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    print("AMP enabled:", use_amp)

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        running_train_loss = 0.0
        running_color_loss = 0.0
        running_chroma_loss = 0.0
        running_edge_loss = 0.0

        # ====================================================
        # Training
        # ====================================================

        for batch_index, (L, target_AB) in enumerate(train_loader):
            L = L.to(device, non_blocking=True)
            target_AB = target_AB.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                predicted_AB = model(L)
                losses = criterion(predicted_AB, target_AB)
                total_loss = losses["total"]

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += total_loss.item()
            running_color_loss += losses["color"].item()
            running_chroma_loss += losses["chroma"].item()
            running_edge_loss += losses["edge"].item()

            if (batch_index + 1) % 500 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Batch {batch_index + 1}/{len(train_loader)} | "
                    f"Loss {total_loss.item():.4f}"
                )

        num_train_batches = len(train_loader)
        train_loss = running_train_loss / num_train_batches
        train_color = running_color_loss / num_train_batches
        train_chroma = running_chroma_loss / num_train_batches
        train_edge = running_edge_loss / num_train_batches

        model.eval()
        running_val_loss = 0.0
        running_val_color = 0.0
        running_val_chroma = 0.0
        running_val_edge = 0.0

        # ====================================================
        # Validation
        # ====================================================

        with torch.no_grad():
            for L, target_AB in validation_loader:
                L = L.to(device, non_blocking=True)
                target_AB = target_AB.to(device, non_blocking=True)

                with autocast(enabled=use_amp):
                    predicted_AB = model(L)
                    losses = criterion(predicted_AB, target_AB)

                running_val_loss += losses["total"].item()
                running_val_color += losses["color"].item()
                running_val_chroma += losses["chroma"].item()
                running_val_edge += losses["edge"].item()

        num_val_batches = len(validation_loader)
        validation_loss = running_val_loss / num_val_batches
        validation_color = running_val_color / num_val_batches
        validation_chroma = running_val_chroma / num_val_batches
        validation_edge = running_val_edge / num_val_batches

        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # Print results
        print()
        print("-" * 70)
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"Training Loss     : {train_loss:.6f}")
        print(f"Training Color    : {train_color:.6f}")
        print(f"Training Chroma   : {train_chroma:.6f}")
        print(f"Training Edge     : {train_edge:.6f}")
        print(f"Validation Loss   : {validation_loss:.6f}")
        print(f"Validation Color  : {validation_color:.6f}")
        print(f"Validation Chroma : {validation_chroma:.6f}")
        print(f"Validation Edge   : {validation_edge:.6f}")
        print(f"Learning Rate     : {current_lr:.8f}")
        print(f"Epoch Time        : {epoch_time:.2f} sec")

        # Save best model
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_validation_loss": best_validation_loss,
                    "num_queries": NUM_QUERIES,
                    "image_size": IMAGE_SIZE
                },
                MODEL_PATH
            )

            print()
            print("BEST MODEL SAVED")
            print("Checkpoint:", MODEL_PATH)

        print("-" * 70)

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print("Best validation loss:", f"{best_validation_loss:.6f}")
    print("Best model:", MODEL_PATH)


if __name__ == "__main__":
    main()