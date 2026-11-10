"""
app_unet.py - Interactive Web Application for U-Net Image Colorization
----------------------------------------------------------------------
Features:
- Built with Gradio for clean, responsive browser UI.
- Loads trained ColorizationUNet weights (defaults to models/unet_colorizer_best.pth).
- Preserves original high-resolution luminance (L channel) for crisp edge retention.
- Predicts chrominance (a*, b*) at network resolution and upsamples back to original size.
- Post-processing HSV Saturation Boost slider for that final vibrant "pop".
- Side-by-side Grayscale vs Colorized preview with download capability.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
import gradio as gr

from unet_model import ColorizationUNet
from advanced_refiner_models import edge_aware_structural_lock


# ==============================================================================
# Configuration & Model Loading
# ==============================================================================

DEFAULT_CHECKPOINT = "models/unet_colorizer_best.pth"
FALLBACK_CHECKPOINTS = [
    "models/unet_colorizer_latest.pth",
]

IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance
model: Optional[ColorizationUNet] = None
current_checkpoint_name: str = "None"


def load_model(checkpoint_path: str = DEFAULT_CHECKPOINT) -> Tuple[ColorizationUNet, str]:
    """Loads ColorizationUNet from the specified checkpoint, falling back if not found or incompatible."""
    global model, current_checkpoint_name

    target_path = checkpoint_path
    if not os.path.isfile(target_path):
        for fallback in FALLBACK_CHECKPOINTS:
            if os.path.isfile(fallback):
                target_path = fallback
                break

    net = ColorizationUNet(in_channels=1, out_channels=2).to(DEVICE)

    if os.path.isfile(target_path):
        print(f"[*] Attempting to load checkpoint from: {target_path}")
        try:
            checkpoint = torch.load(target_path, map_location=DEVICE)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            net.load_state_dict(state_dict)
            status_msg = f"Loaded weights: {os.path.basename(target_path)} ({DEVICE.type.upper()})"
            print(f"[+] {status_msg}")
        except Exception as e:
            status_msg = (
                f"Checkpoint '{os.path.basename(target_path)}' architecture mismatch. "
                f"Running with initialized weights. Please train a new model using train_unet.py."
            )
            print(f"[!] {status_msg} (Error: {e})")
    else:
        status_msg = (
            f"Checkpoint '{checkpoint_path}' not found yet. "
            f"Running with initialized weights. Train model using: python train_unet.py"
        )
        print(f"[!] {status_msg}")

    net.eval()
    model = net
    current_checkpoint_name = target_path
    return net, status_msg


# Initialize model at startup
model, startup_status = load_model(DEFAULT_CHECKPOINT)


# ==============================================================================
# Colorization & Post-Processing Logic
# ==============================================================================

def boost_saturation_hsv(rgb_image: np.ndarray, factor: float = 1.15) -> np.ndarray:
    """
    Applies post-processing saturation boost in HSV color space.
    Gives colorized images that extra crisp 'pop' without altering lightness.
    """
    if abs(factor - 1.0) < 1e-3:
        return rgb_image

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Scale Saturation channel (index 1)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0.0, 255.0)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def colorize(
    input_image: Optional[np.ndarray],
    saturation_boost: float = 1.20,
    custom_checkpoint: str = ""
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Main colorization inference function.
    
    Args:
        input_image: NumPy RGB image array [H, W, 3] from Gradio.
        saturation_boost: Multiplier for HSV saturation (1.0 = unchanged, 1.3 = +30%).
        custom_checkpoint: Optional path to another checkpoint.
    
    Returns:
        (grayscale_view, colorized_rgb, status_message)
    """
    global model, current_checkpoint_name

    if input_image is None:
        return None, None, "Please upload an image to colorize."

    # Reload checkpoint if user specified a custom one that changed
    if custom_checkpoint.strip() and custom_checkpoint.strip() != current_checkpoint_name:
        if os.path.isfile(custom_checkpoint.strip()):
            model, _ = load_model(custom_checkpoint.strip())

    orig_h, orig_w = input_image.shape[:2]

    # Convert RGB -> LAB
    lab = cv2.cvtColor(input_image, cv2.COLOR_RGB2LAB)
    original_L = lab[:, :, 0]  # Full-resolution lightness [0, 255]

    # Resize Lightness channel to network resolution (256x256)
    L_resized = cv2.resize(
        original_L,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Normalize L to [0.0, 1.0]
    L_norm = L_resized.astype(np.float32) / 255.0
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Predict chrominance (a*, b*)
    with torch.inference_mode():
        pred_ab = model(L_tensor)[0].cpu().numpy()  # [2, 256, 256], range [-1.0, 1.0]

    # Upsample predicted chrominance channels back to original image resolution
    pred_a = cv2.resize(
        pred_ab[0],
        (orig_w, orig_h),
        interpolation=cv2.INTER_CUBIC
    )
    pred_b = cv2.resize(
        pred_ab[1],
        (orig_w, orig_h),
        interpolation=cv2.INTER_CUBIC
    )

    # De-normalize [-1, 1] back to OpenCV uint8 [0, 255]
    a_uint8 = np.clip(pred_a * 128.0 + 128.0, 0, 255).astype(np.uint8)
    b_uint8 = np.clip(pred_b * 128.0 + 128.0, 0, 255).astype(np.uint8)

    # Merge with original full-resolution L channel
    colorized_lab = cv2.merge([original_L, a_uint8, b_uint8])
    colorized_rgb = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2RGB)

    # Apply HSV saturation boost
    final_rgb = boost_saturation_hsv(colorized_rgb, factor=saturation_boost)

    # Apply Edge-Aware Structural Locking to eliminate halos and color bleeding
    locked_bgr, _ = edge_aware_structural_lock(
        color_bgr=cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR),
        gray_guide=original_L,
        radius=6,
        eps=0.005
    )
    final_rgb = cv2.cvtColor(locked_bgr, cv2.COLOR_BGR2RGB)

    # Prepare grayscale preview for side-by-side comparison
    gray_preview = cv2.cvtColor(original_L, cv2.COLOR_GRAY2RGB)

    info_msg = (
        f"Success | Resolution: {orig_w}x{orig_h} | "
        f"Device: {DEVICE.type.upper()} | "
        f"Saturation Boost: {saturation_boost:.2f}x"
    )
    return gray_preview, final_rgb, info_msg


# ==============================================================================
# Gradio UI Construction
# ==============================================================================

def create_ui():
    custom_theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="cyan",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=custom_theme, title="AI U-Net Image Colorizer") as demo:
        gr.Markdown(
            """
            # 🎨 AI Image Colorization Studio (U-Net + Perceptual & Chroma Loss)
            Upload a vintage black & white photo or grayscale image. The deep U-Net restores natural,
            vibrant colors with sharp edge fidelity and no sepia washing.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(
                    label="Upload Image (Grayscale or Color)",
                    type="numpy",
                    sources=["upload", "clipboard"]
                )

                saturation_slider = gr.Slider(
                    minimum=0.8,
                    maximum=1.8,
                    value=1.15,
                    step=0.05,
                    label="Saturation Boost (HSV 'Pop')",
                    info="Adjust to intensify colors or soften them (1.0 = neutral output)."
                )

                checkpoint_input = gr.Textbox(
                    label="Model Checkpoint Path",
                    value=DEFAULT_CHECKPOINT,
                    placeholder="e.g. models/unet_colorizer_best.pth",
                    info="Points to your best checkpoint trained with train_unet.py."
                )

                colorize_btn = gr.Button("⚡ Colorize Image", variant="primary", size="lg")
                status_box = gr.Textbox(
                    label="Status",
                    value=startup_status,
                    interactive=False
                )

            with gr.Column(scale=2):
                with gr.Row():
                    out_gray = gr.Image(label="Grayscale Input (Preserved L Channel)", interactive=False)
                    out_color = gr.Image(label="Colorized Result (Vibrant AI Output)", interactive=False)

        gr.Examples(
            examples=[
                ["uploads/image1.jpg", 1.15, DEFAULT_CHECKPOINT],
                ["uploads/image_2.jpeg", 1.20, DEFAULT_CHECKPOINT],
            ],
            inputs=[input_img, saturation_slider, checkpoint_input],
            outputs=[out_gray, out_color, status_box],
            fn=colorize,
            cache_examples=False,
            label="Example Test Images (Click to Test)"
        )

        colorize_btn.click(
            fn=colorize,
            inputs=[input_img, saturation_slider, checkpoint_input],
            outputs=[out_gray, out_color, status_box]
        )

    return demo


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    demo = create_ui()
    # Launch local server
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
