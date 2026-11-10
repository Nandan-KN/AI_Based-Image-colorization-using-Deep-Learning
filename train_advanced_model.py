import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from advanced_dataset_loader import AdvancedColorizationDataset
from cnn_model import CNNColorizationModel


def train_model(data_dir, batch_size=8, image_size=256, epochs=30, learning_rate=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AdvancedColorizationDataset(image_dir=data_dir, image_size=image_size)

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = CNNColorizationModel().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    best_val_loss = float("inf")
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for L_batch, AB_batch in train_loader:
            L_batch = L_batch.to(device, non_blocking=True)
            AB_batch = AB_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_AB = model(L_batch)
            loss = criterion(pred_AB, AB_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * L_batch.size(0)

        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for L_batch, AB_batch in val_loader:
                L_batch = L_batch.to(device, non_blocking=True)
                AB_batch = AB_batch.to(device, non_blocking=True)
                pred_AB = model(L_batch)
                val_loss += criterion(pred_AB, AB_batch).item() * L_batch.size(0)

        val_loss /= len(val_dataset)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                save_dir / "best_model.pt",
            )

    print(f"Training finished. Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="advanced_dataset/images", help="Folder path for the dataset.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )