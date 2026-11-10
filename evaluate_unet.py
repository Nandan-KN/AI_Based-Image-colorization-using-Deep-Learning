"""
evaluate_unet.py - Quantitative & Qualitative Evaluation for U-Net Colorizer
----------------------------------------------------------------------------
Computes:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index Measure)
- MSE / MAE (Mean Squared / Absolute Error)
- Generates side-by-side visual panels:
  [Grayscale Input | U-Net Colorized | HSV Saturation Boosted | Ground Truth]
"""

import os
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn

from unet_model import ColorizationUNet


def calculate_metrics(pred_rgb: np.ndarray, target_rgb: np.ndarray):
    """Computes MSE, MAE, PSNR (dB), and SSIM between predicted and target images."""
    p = pred_rgb.astype(np.float32) / 255.0
    t = target_rgb.astype(np.float32) / 255.0

    diff = p - t
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse < 1e-10 else 10.0 * np.log10(1.0 / mse)
    ssim = float(ssim_fn(t, p, channel_axis=2, data_range=1.0))

    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}


def boost_saturation_hsv(rgb_image: np.ndarray, factor: float = 1.20) -> np.ndarray:
    """Post-processing saturation multiplier in HSV space."""
    if abs(factor - 1.0) < 1e-3:
        return rgb_image
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def colorize_eval_image(model, image_path: str, device: torch.device, image_size: int = 256):
    """Runs inference on a single image and returns grayscale, pred_rgb, boosted_rgb, gt_rgb."""
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    orig_h, orig_w = bgr.shape[:2]
    gt_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2LAB)
    original_L = lab[:, :, 0]

    # Preprocess
    L_small = cv2.resize(original_L, (image_size, image_size), interpolation=cv2.INTER_AREA)
    L_norm = L_small.astype(np.float32) / 255.0
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        pred_ab = model(L_tensor)[0].cpu().numpy()

    # Upsample
    pred_a = cv2.resize(pred_ab[0], (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    pred_b = cv2.resize(pred_ab[1], (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    a_uint8 = np.clip(pred_a * 128.0 + 128.0, 0, 255).astype(np.uint8)
    b_uint8 = np.clip(pred_b * 128.0 + 128.0, 0, 255).astype(np.uint8)

    colorized_lab = cv2.merge([original_L, a_uint8, b_uint8])
    pred_rgb = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2RGB)
    boosted_rgb = boost_saturation_hsv(pred_rgb, factor=1.20)
    gray_rgb = cv2.cvtColor(original_L, cv2.COLOR_GRAY2RGB)

    return gray_rgb, pred_rgb, boosted_rgb, gt_rgb


def create_comparison_panel(panels, target_height: int = 384) -> np.ndarray:
    """Concatenates panels horizontally with header labels."""
    header_h = 36
    resized = []
    for label, img in panels:
        h, w = img.shape[:2]
        new_w = int(w * (target_height / h))
        img_res = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_height + header_h, new_w, 3), dtype=np.uint8)
        canvas[header_h:, :] = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
        cv2.putText(
            canvas, label, (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
        )
        resized.append(canvas)
    return np.hstack(resized)


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation Device: {device}")

    # Load Model
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = ColorizationUNet(in_channels=1, out_channels=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[+] Loaded weights: {args.checkpoint}")

    # Collect Images
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(args.data_dir, "**", ext), recursive=True))
        paths.extend(glob.glob(os.path.join(args.data_dir, ext)))
    paths = sorted(list(set(paths)))

    if not paths:
        raise FileNotFoundError(f"No images found in {args.data_dir}")

    eval_paths = paths[:args.num_samples] if args.num_samples > 0 else paths
    print(f"[*] Evaluating {len(eval_paths)} images...")

    os.makedirs(args.output_dir, exist_ok=True)
    all_metrics = {"mse": [], "mae": [], "psnr": [], "ssim": []}

    for idx, path in enumerate(eval_paths):
        res = colorize_eval_image(model, path, device, args.image_size)
        if res is None:
            continue
        gray_rgb, pred_rgb, boosted_rgb, gt_rgb = res

        m = calculate_metrics(pred_rgb, gt_rgb)
        for k in all_metrics:
            all_metrics[k].append(m[k])

        if idx < args.save_panels:
            panel = create_comparison_panel([
                ("Grayscale Input", gray_rgb),
                ("U-Net (Raw)", pred_rgb),
                ("U-Net (+HSV Pop)", boosted_rgb),
                ("Ground Truth", gt_rgb)
            ])
            panel_filename = os.path.join(args.output_dir, f"eval_panel_{idx+1:03d}_{Path(path).stem}.jpg")
            cv2.imwrite(panel_filename, panel)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(eval_paths):
            print(f"[{idx+1:03d}/{len(eval_paths):03d}] Avg PSNR: {np.mean(all_metrics['psnr']):.2f} dB | Avg SSIM: {np.mean(all_metrics['ssim']):.4f}")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Images Evaluated : {len(all_metrics['mse'])}")
    print(f"Mean Squared Error (MSE) : {np.mean(all_metrics['mse']):.6f}")
    print(f"Mean Absolute Error (MAE): {np.mean(all_metrics['mae']):.6f}")
    print(f"PSNR (dB)                : {np.mean(all_metrics['psnr']):.2f} dB")
    print(f"SSIM                     : {np.mean(all_metrics['ssim']):.4f}")
    print(f"Visual panels saved to   : {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate U-Net Colorization Model")
    parser.add_argument("--checkpoint", type=str, default="models/resnet18_chroma_epoch1.pth", help="Checkpoint path")
    parser.add_argument("--data_dir", type=str, default="uploads", help="Directory with evaluation images")
    parser.add_argument("--output_dir", type=str, default="outputs/unet_eval", help="Directory for output panels")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of images to evaluate (0 for all)")
    parser.add_argument("--save_panels", type=int, default=5, help="Number of visual comparison panels to save")
    parser.add_argument("--image_size", type=int, default=256, help="Inference resolution")
    args = parser.parse_args()
    run_evaluation(args)
