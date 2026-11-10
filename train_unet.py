"""
train_unet.py - Complete Training Pipeline for U-Net Image Colorization
-----------------------------------------------------------------------
Key Highlights:
1. Data Handling:
   - Robust loading of 20,000+ images from any folder structure.
   - Converts RGB images to CIELAB color space via OpenCV.
   - Extracts Lightness 'L' in [0, 1] as network input.
   - Extracts chrominance 'a*' and 'b*' normalized to [-1, 1] as targets.

2. Multi-Objective Loss Function (CRITICAL):
   - Pixel Accuracy Loss: L1 Loss between predicted (a, b) and ground truth (a, b).
   - Perceptual Loss: Pre-trained VGG16 feature loss using fully differentiable
     PyTorch CIELAB -> RGB conversion, allowing gradients to flow back into U-Net.
   - Chroma Loss: Penalizes low-saturation/washed-out/sepia outputs by enforcing
     chromatic magnitude sqrt(a^2 + b^2).

3. Training:
   - Adam optimizer with ReduceLROnPlateau scheduler.
   - Checkpointing (best and latest).
"""

import os
import glob
import random
import argparse
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models

from unet_model import ColorizationUNet


# ==============================================================================
# 1. Differentiable CIELAB -> RGB Converter (PyTorch)
# ==============================================================================

class DifferentiableLabToRGB(nn.Module):
    """
    Differentiable Lab -> RGB conversion implemented strictly in PyTorch.
    Crucial for computing Perceptual Loss with VGG backpropagating into U-Net.

    Inputs:
        L  : Lightness channel [B, 1, H, W] in [0.0, 1.0]
        ab : Chrominance channels [B, 2, H, W] in [-1.0, 1.0]
    Output:
        RGB: Color image [B, 3, H, W] in [0.0, 1.0]
    """
    def __init__(self):
        super().__init__()
        # Standard CIE D65 illuminant white point constants
        self.register_buffer("Xn", torch.tensor(0.95047))
        self.register_buffer("Yn", torch.tensor(1.00000))
        self.register_buffer("Zn", torch.tensor(1.08883))
        self.delta = 6.0 / 29.0

    def forward(self, L: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
        # Scale back to standard CIELAB ranges
        # L in [0, 100], a in [-128, 128], b in [-128, 128]
        L_scaled = L * 100.0
        a_scaled = ab[:, 0:1, :, :] * 128.0
        b_scaled = ab[:, 1:2, :, :] * 128.0

        # Lab -> XYZ intermediate functions
        fy = (L_scaled + 16.0) / 116.0
        fx = (a_scaled / 500.0) + fy
        fz = fy - (b_scaled / 200.0)

        # Non-linear function inverse
        def finv(t: torch.Tensor) -> torch.Tensor:
            return torch.where(
                t > self.delta,
                t ** 3,
                3.0 * (self.delta ** 2) * (t - 4.0 / 29.0)
            )

        X = finv(fx) * self.Xn
        Y = finv(fy) * self.Yn
        Z = finv(fz) * self.Zn

        # XYZ -> Linear sRGB transformation matrix
        r_lin =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
        g_lin = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
        b_lin =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

        rgb_lin = torch.cat([r_lin, g_lin, b_lin], dim=1)
        rgb_lin = torch.clamp(rgb_lin, min=1e-6)

        # Gamma companding for sRGB
        rgb = torch.where(
            rgb_lin > 0.0031308,
            1.055 * (rgb_lin ** (1.0 / 2.4)) - 0.055,
            12.92 * rgb_lin
        )
        return torch.clamp(rgb, 0.0, 1.0)


# ==============================================================================
# 2. Combined Loss Function (L1 + Perceptual VGG + Chroma Loss)
# ==============================================================================

class VGGPerceptualLoss(nn.Module):
    """
    Computes perceptual feature difference between predicted and target images
    using multi-scale feature maps from pre-trained VGG16 (relu1_2, relu2_2, relu3_3).
    """
    def __init__(self, device: torch.device):
        super().__init__()
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        except Exception:
            vgg = models.vgg16(pretrained=True).features

        # Extract features up to relu1_2 (index 4), relu2_2 (index 9), relu3_3 (index 16)
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(4)]).to(device).eval()
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)]).to(device).eval()
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 16)]).to(device).eval()

        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization parameters
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        # Normalize to standard ImageNet statistics
        p = (pred_rgb - self.mean) / self.std
        t = (target_rgb - self.mean) / self.std

        # Extract features
        p1 = self.slice1(p); t1 = self.slice1(t)
        p2 = self.slice2(p1); t2 = self.slice2(t1)
        p3 = self.slice3(p2); t3 = self.slice3(t2)

        loss = (
            F.l1_loss(p1, t1)
            + 0.8 * F.l1_loss(p2, t2)
            + 0.5 * F.l1_loss(p3, t3)
        )
        return loss


class ChromaLoss(nn.Module):
    """
    Penalizes desaturated/washed-out outputs.
    Chroma in LAB is defined as sqrt(a^2 + b^2).
    This loss computes:
    1. L1 difference between predicted chroma and target chroma.
    2. An under-saturation penalty when predicted chroma is significantly
       lower than ground truth.
    """
    def __init__(self, undersaturation_penalty: float = 0.5):
        super().__init__()
        self.undersaturation_penalty = undersaturation_penalty

    def forward(self, pred_ab: torch.Tensor, target_ab: torch.Tensor) -> torch.Tensor:
        pred_chroma = torch.sqrt(pred_ab[:, 0:1] ** 2 + pred_ab[:, 1:2] ** 2 + 1e-6)
        target_chroma = torch.sqrt(target_ab[:, 0:1] ** 2 + target_ab[:, 1:2] ** 2 + 1e-6)

        # Pixel-wise chroma difference
        chroma_l1 = F.l1_loss(pred_chroma, target_chroma)

        # Under-saturation penalty: heavily penalize model if it collapses to gray
        undersat_loss = torch.mean(F.relu(target_chroma - pred_chroma))

        return chroma_l1 + (self.undersaturation_penalty * undersat_loss)


class ColorizationLoss(nn.Module):
    """
    Combined high-fidelity colorization loss:
    Loss = lambda_l1 * L1 + lambda_vgg * Perceptual_VGG + lambda_chroma * Chroma
    """
    def __init__(
        self,
        device: torch.device,
        lambda_l1: float = 1.0,
        lambda_vgg: float = 0.25,
        lambda_chroma: float = 0.5,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_chroma = lambda_chroma

        self.lab_to_rgb = DifferentiableLabToRGB().to(device)
        self.vgg_loss = VGGPerceptualLoss(device)
        self.chroma_loss = ChromaLoss(undersaturation_penalty=0.5).to(device)

    def forward(
        self,
        pred_ab: torch.Tensor,
        target_ab: torch.Tensor,
        L_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        # 1. L1 loss on (a, b) channels
        l1_loss = F.l1_loss(pred_ab, target_ab)

        # 2. Chroma loss to avoid sepia/washed-out outputs
        chroma_loss = self.chroma_loss(pred_ab, target_ab)

        # 3. Differentiable RGB conversion for Perceptual VGG loss
        pred_rgb = self.lab_to_rgb(L_input, pred_ab)
        target_rgb = self.lab_to_rgb(L_input, target_ab)
        vgg_loss = self.vgg_loss(pred_rgb, target_rgb)

        total_loss = (
            self.lambda_l1 * l1_loss
            + self.lambda_vgg * vgg_loss
            + self.lambda_chroma * chroma_loss
        )

        loss_breakdown = {
            "total": total_loss.item(),
            "l1": l1_loss.item(),
            "vgg": vgg_loss.item(),
            "chroma": chroma_loss.item(),
        }
        return total_loss, loss_breakdown


# ==============================================================================
# 3. Dataset & Data Loader
# ==============================================================================

class ColorizationDataset(Dataset):
    """
    PyTorch Dataset for CIELAB Image Colorization.
    Loads standard RGB images, resizes, performs data augmentation,
    converts to CIELAB with OpenCV, and prepares normalized tensors.
    """
    def __init__(self, image_paths: List[str], image_size: int = 256, is_training: bool = True):
        self.image_paths = image_paths
        self.image_size = image_size
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if bgr is None:
            # Fallback for corrupted file: generate neutral image
            bgr = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 128

        # Resize
        bgr = cv2.resize(bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        # Data augmentation (Random Horizontal Flip) during training
        if self.is_training and random.random() > 0.5:
            bgr = cv2.flip(bgr, 1)

        # Convert BGR -> RGB -> LAB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

        # Channel extraction:
        # L in OpenCV is [0, 255] -> normalize to [0.0, 1.0]
        L = lab[:, :, 0].astype(np.float32) / 255.0

        # a, b in OpenCV are [0, 255] (centered around 128) -> normalize to [-1.0, 1.0]
        a = (lab[:, :, 1].astype(np.float32) - 128.0) / 128.0
        b = (lab[:, :, 2].astype(np.float32) - 128.0) / 128.0

        # Tensors: L: [1, H, W], ab: [2, H, W]
        L_tensor = torch.from_numpy(L).unsqueeze(0)
        ab_tensor = torch.from_numpy(np.stack([a, b], axis=0))

        return L_tensor, ab_tensor


def get_image_paths(data_dir: str) -> List[str]:
    """Recursively search for supported image formats in data_dir."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        paths.extend(glob.glob(os.path.join(data_dir, ext)))
    return sorted(list(set(paths)))


# ==============================================================================
# 4. Training Loop
# ==============================================================================

def train_colorizer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")
    if device.type == "cuda":
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Collect Dataset
    print(f"[*] Scanning images from: {args.data_dir}")
    all_images = get_image_paths(args.data_dir)
    total_found = len(all_images)
    print(f"[*] Found {total_found} images.")

    if total_found == 0:
        raise FileNotFoundError(f"No valid images found in directory: {args.data_dir}")

    # Train / Validation split (90% train, 10% val)
    random.seed(args.seed)
    random.shuffle(all_images)
    val_size = max(1, int(len(all_images) * args.val_split))
    train_paths = all_images[:-val_size]
    val_paths = all_images[-val_size:]

    print(f"[*] Training samples  : {len(train_paths)}")
    print(f"[*] Validation samples: {len(val_paths)}")

    train_dataset = ColorizationDataset(train_paths, image_size=args.image_size, is_training=True)
    val_dataset = ColorizationDataset(val_paths, image_size=args.image_size, is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 2. Model, Loss, Optimizer, and LR Scheduler
    model = ColorizationUNet(in_channels=1, out_channels=2, base_filters=args.base_filters).to(device)
    criterion = ColorizationLoss(
        device=device,
        lambda_l1=args.lambda_l1,
        lambda_vgg=args.lambda_vgg,
        lambda_chroma=args.lambda_chroma
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        verbose=True
    )

    # Automatic Mixed Precision (AMP) to optimize speed and fit 4GB GPU VRAM
    use_amp = (device.type == "cuda" and not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[*] Automatic Mixed Precision (AMP FP16) ENABLED for fast GPU training.")

    # Resume if checkpoint specified
    start_epoch = 1
    best_val_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)

    if args.resume and os.path.isfile(args.resume):
        print(f"[*] Resuming checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"[*] Resumed at epoch {start_epoch} (best val loss: {best_val_loss:.4f})")

    # 3. Training Epochs
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_l1 = 0.0
        train_vgg = 0.0
        train_chroma = 0.0

        for step, (L_batch, ab_batch) in enumerate(train_loader):
            L_batch = L_batch.to(device)
            ab_batch = ab_batch.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_ab = model(L_batch)
                loss, breakdown = criterion(pred_ab, ab_batch, L_batch)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += breakdown["total"]
            train_l1 += breakdown["l1"]
            train_vgg += breakdown["vgg"]
            train_chroma += breakdown["chroma"]

            if (step + 1) % args.log_interval == 0 or (step + 1) == len(train_loader):
                print(
                    f"Epoch [{epoch:02d}/{args.epochs:02d}] "
                    f"Step [{step + 1:04d}/{len(train_loader):04d}] "
                    f"Loss: {loss.item():.4f} "
                    f"(L1: {breakdown['l1']:.4f}, VGG: {breakdown['vgg']:.4f}, Chroma: {breakdown['chroma']:.4f})"
                )

        train_loss /= len(train_loader)
        train_l1 /= len(train_loader)
        train_vgg /= len(train_loader)
        train_chroma /= len(train_loader)

        # 4. Validation Loop
        model.eval()
        val_loss = 0.0
        val_l1 = 0.0

        with torch.no_grad():
            for L_val, ab_val in val_loader:
                L_val = L_val.to(device)
                ab_val = ab_val.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred_val_ab = model(L_val)
                    v_loss, v_breakdown = criterion(pred_val_ab, ab_val, L_val)
                val_loss += v_breakdown["total"]
                val_l1 += v_breakdown["l1"]

        val_loss /= len(val_loader)
        val_l1 /= len(val_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print("=" * 70)
        print(
            f"--> Epoch {epoch:02d} Summary | Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} (Val L1: {val_l1:.4f}) | LR: {current_lr:.6f}"
        )
        print("=" * 70)

        # Save latest checkpoint
        latest_path = os.path.join(args.save_dir, "unet_colorizer_latest.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        }, latest_path)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, "unet_colorizer_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            }, best_path)
            print(f"[+] Saved NEW BEST model to: {best_path} (Val Loss: {best_val_loss:.4f})")

    print("\n[+] Training finished successfully!")


# ==============================================================================
# 5. CLI Arguments
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for Image Colorization with VGG & Chroma Loss")
    parser.add_argument("--data_dir", type=str, default="advanced_dataset/images", help="Path to image dataset folder")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (8 is optimal for 4GB VRAM)")
    parser.add_argument("--image_size", type=int, default=256, help="Image resolution (H=W=256)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--base_filters", type=int, default=64, help="Base filter count for U-Net")
    parser.add_argument("--val_split", type=float, default=0.10, help="Validation fraction (e.g. 0.10 for 10 percent)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--lambda_l1", type=float, default=1.0, help="Weight for L1 pixel loss")
    parser.add_argument("--lambda_vgg", type=float, default=0.25, help="Weight for Perceptual VGG loss")
    parser.add_argument("--lambda_chroma", type=float, default=0.5, help="Weight for Chroma saturation loss")
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision FP16")
    parser.add_argument("--log_interval", type=int, default=50, help="Step interval for batch logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resume", type=str, default="", help="Optional path to checkpoint to resume")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_colorizer(args)
