"""
restore_image_clean.py
======================
Generates clean, realistic, vibrant colorization (matching Image 2)
while eliminating all psychedelic distortions, color halos, and channel swaps (Image 1).

Key Pipeline Steps:
1. Strict BGR -> RGB -> LAB color conversion (prevents red/blue axis inversion).
2. Deep color inference using our Teacher-Guided DDColor or Champion ResNet-18.
3. Native full-resolution luminance retention (preserves 100% of photographic sharpness).
4. Edge-Aware Structural Locking via Joint Guided Filter (eliminates halos/bleeding).
5. Controlled, balanced saturation pop (prevents neon pixel clipping).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from advanced_ddcolor_model import AdvancedDDColorInspired
from advanced_resnet_colorizer import ResNet18Colorizer
from advanced_refiner_models import edge_aware_structural_lock


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def restore_clean(
    image_path: str,
    output_path: str,
    model_type: str = "teacher",
    saturation_boost: float = 1.15,
    enable_edge_lock: bool = True
):
    print(f"[*] Reading image: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    orig_h, orig_w = image_bgr.shape[:2]
    print(f"[*] Native Resolution: {orig_w}x{orig_h} | Target Model: {model_type.upper()}")

    # 1. Color Conversion (Strict BGR -> RGB -> LAB to avoid channel flip)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    native_L = lab[:, :, 0]

    # 2. Resize L channel for neural processing (256x256)
    L_small = cv2.resize(native_L, (256, 256), interpolation=cv2.INTER_AREA)
    L_norm = L_small.astype(np.float32) / 255.0
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # 3. Model Inference
    if model_type.lower() == "teacher":
        ckpt_path = "models/advanced_ddcolor_teacher_w15_epoch1.pth"
        model = AdvancedDDColorInspired(num_queries=16).to(DEVICE)
    else:
        ckpt_path = "models/resnet18_chroma_epoch1.pth"
        model = ResNet18Colorizer(pretrained=False).to(DEVICE)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()

    with torch.inference_mode():
        pred_ab = model(L_tensor)[0].cpu().numpy()

    # 4. Upsample predicted chrominance to native resolution
    # Index 0 = a* (green-red), Index 1 = b* (blue-yellow)
    pred_a = cv2.resize(pred_ab[0], (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    pred_b = cv2.resize(pred_ab[1], (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Correct denormalization: center around 128
    a_uint8 = np.clip(pred_a * 128.0 + 128.0, 0, 255).astype(np.uint8)
    b_uint8 = np.clip(pred_b * 128.0 + 128.0, 0, 255).astype(np.uint8)

    # 5. Merge with native high-res L channel
    restored_lab = cv2.merge([native_L, a_uint8, b_uint8])
    restored_rgb = cv2.cvtColor(restored_lab, cv2.COLOR_LAB2RGB)

    # 6. Controlled Saturation Pop (HSV space)
    if saturation_boost != 1.0:
        hsv = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0.0, 255.0)
        restored_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)

    # 7. Edge-Aware Structural Locking (Eliminates halos and color bleeding)
    if enable_edge_lock:
        print("[*] Applying Edge-Aware Structural Locking (Joint Guided Filter)...")
        final_bgr, _ = edge_aware_structural_lock(
            color_bgr=restored_bgr,
            gray_guide=native_L,
            radius=6,
            eps=0.005
        )
    else:
        final_bgr = restored_bgr

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, final_bgr)
    print(f"[+] Successfully saved clean colorized image to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore clean, realistic colorization like Image 2")
    parser.add_argument("--input", type=str, default="uploads_resnet/5441579a6a6d418aa5b2f69d70198176_image12.jpg", help="Path to input photo")
    parser.add_argument("--output", type=str, default="outputs/image12_clean_like_image2.jpg", help="Path to output photo")
    parser.add_argument("--model", type=str, default="teacher", choices=["teacher", "resnet"], help="Model backbone")
    parser.add_argument("--saturation", type=float, default=1.15, help="Saturation boost (1.0 to 1.25)")
    parser.add_argument("--no_edge_lock", action="store_true", help="Disable anti-bleeding edge filter")
    args = parser.parse_args()

    restore_clean(
        image_path=args.input,
        output_path=args.output,
        model_type=args.model,
        saturation_boost=args.saturation,
        enable_edge_lock=not args.no_edge_lock
    )
