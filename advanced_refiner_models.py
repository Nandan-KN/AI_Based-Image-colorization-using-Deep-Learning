"""
advanced_refiner_models.py
==========================
Custom PyTorch Neural Models and Edge-Constrained Guided Processing for:
1. ScribbleRefinerCNN (Human-in-the-loop Stroke Fusion Network)
2. CLIPColorAdapter (Text-Conditioned Semantic Color Modulation)
3. Edge-Aware Structural Locking (Anti-Bleeding Guided Filter Engine)
"""

import math
from typing import Tuple, Optional, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Edge-Aware Structural Locking (Joint Guided Filtering & Anti-Bleed)
# ==============================================================================

def guided_filter_mono(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-2) -> np.ndarray:
    """
    Memory-efficient Fast Guided Filter (He and Sun, ECCV 2015).
    Safely handles ultra-high resolution images (4K, 8K+) without ArrayMemoryError.
    """
    h, w = guide.shape[:2]
    max_dim = max(h, w)

    # Subsampling factor for huge images (caps intermediate computation to ~1280px)
    if max_dim > 1280:
        s = int(np.ceil(max_dim / 1280.0))
        small_w = max(1, w // s)
        small_h = max(1, h // s)
        small_guide = cv2.resize(guide, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small_src = cv2.resize(src, (small_w, small_h), interpolation=cv2.INTER_AREA)
        r_small = max(1, radius // s)
    else:
        s = 1
        small_guide = guide
        small_src = src
        r_small = radius

    ksize = (2 * r_small + 1, 2 * r_small + 1)
    mean_I = cv2.boxFilter(small_guide, -1, ksize)
    mean_p = cv2.boxFilter(small_src, -1, ksize)
    mean_Ip = cv2.boxFilter(small_guide * small_src, -1, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(small_guide * small_guide, -1, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)

    if s > 1:
        mean_a = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)

    return mean_a * guide + mean_b


def edge_aware_structural_lock(
    color_bgr: np.ndarray,
    gray_guide: np.ndarray,
    radius: int = 6,
    eps: float = 0.005,
    canny_thresh1: int = 50,
    canny_thresh2: int = 150,
    edge_blend: float = 0.85
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Edge-Aware Structural Locking Post-Processor:
    1. Detects Canny structural edges on the pristine grayscale guide image.
    2. Applies Joint Guided Filtering on CIELAB chrominance channels (a*, b*)
       using the grayscale luminance as the structural barrier.
    3. Locks colors from bleeding across contours (e.g., blue sky bleeding into tree branches).

    Returns:
        (locked_color_bgr, canny_edge_vis)
    """
    orig_h, orig_w = gray_guide.shape[:2]

    # Step 1: Detect Canny edges on original luminance
    edges = cv2.Canny(gray_guide, canny_thresh1, canny_thresh2)

    # Step 2: Convert color image to LAB
    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    L_chan = lab[:, :, 0]
    A_chan = lab[:, :, 1].astype(np.float32) / 255.0
    B_chan = lab[:, :, 2].astype(np.float32) / 255.0

    guide_norm = (gray_guide.astype(np.float32) / 255.0)

    # Step 3: Run Guided Filter independently on A and B channels using L as guide
    A_filtered = guided_filter_mono(guide_norm, A_chan, radius=radius, eps=eps)
    B_filtered = guided_filter_mono(guide_norm, B_chan, radius=radius, eps=eps)

    # Step 4: Blend filtered chrominance back with original
    A_final = (edge_blend * A_filtered + (1.0 - edge_blend) * A_chan)
    B_final = (edge_blend * B_filtered + (1.0 - edge_blend) * B_chan)

    A_uint8 = np.clip(A_final * 255.0, 0, 255).astype(np.uint8)
    B_uint8 = np.clip(B_final * 255.0, 0, 255).astype(np.uint8)

    # Step 5: Re-merge with original full-res L channel
    locked_lab = cv2.merge([L_chan, A_uint8, B_uint8])
    locked_bgr = cv2.cvtColor(locked_lab, cv2.COLOR_LAB2BGR)

    # Edge visualization map
    edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return locked_bgr, edge_vis


# ==============================================================================
# 2. Scribble-Guided Color Diffusion Engine (Affinity Flood-Fill)
# ==============================================================================

def diffuse_scribbles_on_luminance(
    gray_image: np.ndarray,
    scribble_rgb: np.ndarray,
    scribble_mask: np.ndarray,
    iterations: int = 12,
    gradient_sens: float = 0.08
) -> np.ndarray:
    """
    Diffuses user strokes across adjacent pixels having similar grayscale luminance,
    effectively propagating a stroke across an entire semantic object (e.g., an entire
    car or the sky) while halting diffusion at strong luminance edges.
    """
    h, w = gray_image.shape[:2]
    gray_f = gray_image.astype(np.float32) / 255.0
    scribble_f = scribble_rgb.astype(np.float32) / 255.0
    mask_f = (scribble_mask > 0).astype(np.float32)[:, :, 0:1]

    # Compute luminance spatial gradients (Sobel)
    grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Affinity weight: high where gradient is smooth, low at edges
    affinity = np.exp(-grad_mag / (gradient_sens + 1e-5))
    affinity = np.expand_dims(affinity, axis=-1)

    propagated = scribble_f.copy()
    current_mask = mask_f.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for _ in range(iterations):
        # Dilate current mask
        dilated_mask = cv2.dilate(current_mask, kernel)[:, :, None]
        new_frontier = np.clip(dilated_mask - current_mask, 0.0, 1.0)

        # Blur propagated colors to neighboring frontier
        blurred_colors = cv2.GaussianBlur(propagated, (9, 9), 0)

        # Update frontier pixels modulated by local luminance affinity
        update_weight = new_frontier * affinity
        propagated = (1.0 - update_weight) * propagated + update_weight * blurred_colors
        current_mask = np.clip(current_mask + update_weight, 0.0, 1.0)

        # Keep original scribbles permanently pinned
        propagated = (1.0 - mask_f) * propagated + mask_f * scribble_f

    return np.clip(propagated * 255.0, 0, 255).astype(np.uint8)


# ==============================================================================
# 3. Custom PyTorch Refiner CNN (Scribble + Base Colorization Fusion)
# ==============================================================================

class DilatedResidualBlock(nn.Module):
    """
    Residual Block with Dilated Convolutions. Expands the receptive field
    to capture large semantic regions (like an entire car or sky) without
    losing fine textural resolution.
    """
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ScribbleRefinerCNN(nn.Module):
    """
    Custom Refiner Network (Human-in-the-Loop Fusion):
    Inputs:
      1. Grayscale luminance L (1 channel)
      2. DDColor baseline prediction (3 channels, RGB)
      3. User scribble image S (3 channels, RGB)
      4. User scribble mask M (1 channel, binary)
      Total: 8 channels

    Outputs:
      1. Spatially-adaptive blending gate W (1 channel in [0, 1])
      2. Residual chromatic refinement (3 channels in [-1, 1])
      Final Refined Image = (1 - W) * DDColor + W * Scribble_Propagated + Residual
    """
    def __init__(self, in_channels: int = 8, hidden_dim: int = 64):
        super().__init__()

        # Input projection
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Multi-scale dilated residual backbone
        self.res1 = DilatedResidualBlock(hidden_dim, dilation=1)
        self.res2 = DilatedResidualBlock(hidden_dim, dilation=2)
        self.res3 = DilatedResidualBlock(hidden_dim, dilation=4)
        self.res4 = DilatedResidualBlock(hidden_dim, dilation=1)

        # Gating Head: Predicts how much to trust user strokes vs DDColor baseline
        self.gate_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Range [0.0, 1.0]
        )

        # Color Refinement Head: Predicts continuous RGB chromatic adjustments
        self.refine_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh()  # Residual in [-1.0, 1.0]
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        gray: torch.Tensor,
        base_color: torch.Tensor,
        scribble: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gray: [B, 1, H, W] in [0, 1]
            base_color: [B, 3, H, W] in [0, 1]
            scribble: [B, 3, H, W] in [0, 1]
            mask: [B, 1, H, W] in {0, 1}
        Returns:
            fused_image: [B, 3, H, W] in [0, 1]
            blending_gate: [B, 1, H, W]
        """
        # Stack inputs: 1 + 3 + 3 + 1 = 8 channels
        x = torch.cat([gray, base_color, scribble, mask], dim=1)

        feat = self.conv_in(x)
        feat = self.res1(feat)
        feat = self.res2(feat)
        feat = self.res3(feat)
        feat = self.res4(feat)

        gate = self.gate_head(feat)
        residual = self.refine_head(feat) * 0.15  # Scaled residual

        # Blend: Keep DDColor where no scribble exists, adopt user color where scribbled
        blended = (1.0 - gate) * base_color + gate * scribble + residual
        fused = torch.clamp(blended, 0.0, 1.0)

        return fused, gate


# ==============================================================================
# 4. Custom CLIP Text-Prompted Color Adapter
# ==============================================================================

class CLIPColorAdapter(nn.Module):
    """
    Multimodal CLIP Color Modulation Adapter:
    Takes a 512-dimensional CLIP text embedding (from 'openai/clip-vit-base-patch32')
    and projects it into spatial color modulation vectors:
    1. Global Hue & Saturation shift vector
    2. LAB (a*, b*) offset directions
    3. Channel-wise FiLM (Feature-wise Linear Modulation) parameters
    """
    def __init__(self, clip_dim: int = 512, hidden_dim: int = 256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6)  # Output: [delta_a, delta_b, scale_a, scale_b, sat_boost, hue_rot]
        )

        # Semantic Palette Dictionary (fallback embeddings for common color prompts)
        self.preset_vectors = {
            "sunset": torch.tensor([0.45, 0.50, 1.25, 1.20, 1.35, 0.05]),
            "vintage red car": torch.tensor([0.65, 0.25, 1.30, 1.10, 1.40, -0.02]),
            "autumn golden leaves": torch.tensor([0.25, 0.60, 1.15, 1.40, 1.30, 0.08]),
            "cyberpunk neon": torch.tensor([-0.40, -0.55, 1.40, 1.45, 1.60, 0.20]),
            "emerald green": torch.tensor([-0.55, 0.40, 1.30, 1.20, 1.30, -0.10]),
            "ocean blue": torch.tensor([-0.20, -0.65, 1.10, 1.40, 1.30, 0.15]),
            "rose pink": torch.tensor([0.55, -0.15, 1.30, 1.15, 1.35, -0.05]),
        }

    def forward(self, text_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_embedding: [B, 512] float tensor
        Returns:
            Dictionary containing modulation parameters.
        """
        params = self.mlp(text_embedding)  # [B, 6]
        return {
            "delta_a": params[:, 0:1],     # Shift in green-red
            "delta_b": params[:, 1:2],     # Shift in blue-yellow
            "scale_a": torch.sigmoid(params[:, 2:3]) * 1.5 + 0.5,
            "scale_b": torch.sigmoid(params[:, 3:4]) * 1.5 + 0.5,
            "sat_boost": torch.sigmoid(params[:, 4:5]) * 1.5 + 0.8,
            "hue_rot": torch.tanh(params[:, 5:6]) * 0.2
        }

    def get_fallback_vector(self, prompt: str) -> torch.Tensor:
        """Computes rule-based semantic color vector if CLIP model is offline."""
        prompt_lower = prompt.lower()
        matched = []
        for key, vec in self.preset_vectors.items():
            if any(w in prompt_lower for w in key.split()):
                matched.append(vec)

        if matched:
            return torch.mean(torch.stack(matched), dim=0, keepdim=True)
        else:
            # Default mild saturation pop
            return torch.tensor([[0.0, 0.0, 1.1, 1.1, 1.2, 0.0]])
