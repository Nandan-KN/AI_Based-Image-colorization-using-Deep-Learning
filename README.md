# AI-Based Automatic Image Colorization Using Deep Learning and DDColor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.0+-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning framework for photorealistic, real-time image colorization. Operates in **CIE $L^*a^*b^*$ color space** to preserve 100% native photograph sharpness while synthesizing vibrant, natural chrominance.

---

## 🌟 Key Features

- **CIE $L^*a^*b^*$ Space Synthesis**: Decouples lightness ($L^*$) from chrominance ($a^*, b^*$), ensuring zero loss of resolution, texture, or fine edge detail.
- **Sub-Second Real-Time Inference**: Generates colorized output in **220ms – 570ms** on standard CPU hardware.
- **Photographic Vibrance Engine**: Eliminates the "sepia wash" and "regression-to-the-mean" through multiplicative non-linear vibrance.
- **Subsampled Joint Guided Filter**: Locks color transitions strictly to structural contours in $<15\text{ms}$, preventing color bleeding.
- **Unified Multi-Model Architecture**:
  - **CNN Model** (`models/cnn_best.pth`): 4-stage convolutional encoder-decoder with batch normalization (~1.8 MB).
  - **DDColor** (`models/advanced_ddcolor_best.pth`): Dual-decoder architecture with 16 learnable color query embeddings (~29 MB).
  - **ResNet-18 Chroma**: Deep residual skip-connection network.
- **Modern Web Interface**: Built with Gradio Blocks featuring live latency telemetry and high-resolution side-by-side preview.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Nandan-KN/AI-Image-Colorization-DDColor.git
cd AI-Image-Colorization-DDColor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
python app_advanced_master.py
```
Open **`http://127.0.0.1:7860`** in your web browser.

---

## 📊 Quantitative Benchmarks (COCO Test Set)

| Model Architecture | MSE ($\downarrow$) | MAE ($\downarrow$) | PSNR (dB) ($\uparrow$) | SSIM ($\uparrow$) | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 Chroma** | 0.0077 | 0.0531 | 23.14 dB | 0.9023 | **~228 ms** |
| **DDColor (Dual-Decoder)** | 0.0079 | 0.0541 | 23.05 dB | 0.8980 | **~381 ms** |
| **Baseline CNN** | **0.0076** | 0.0527 | 23.15 dB | 0.8999 | **~574 ms** |
| **U-Net (Standard)** | 0.0077 | **0.0525** | **23.24 dB** | **0.9042** | ~680 ms |

---

## 🏗️ System Pipeline

```
[Grayscale Input Image]
         │
         ▼
[1. Extract Native L Channel] ─── (Preserves 100% native sharpness)
         │
         ▼
[2. Neural Inference (CNN / DDColor)] ─── (Predicts normalized a, b channels)
         │
         ▼
[3. Fast Bilinear Upsampling]
         │
         ▼
[4. O(1) Photographic Luminance Mask] ─── (Protects white skies & deep black shadows)
         │
         ▼
[5. Multiplicative Photographic Vibrance] ─── (S_new = S * (1 + vib * (1 - S)))
         │
         ▼
[6. Subsampled Joint Guided Filter] ─── (Locks color edges to physical contours)
         │
         ▼
[Final Colorized RGB Image]
```

---

## 📁 Repository Structure

```
├── app_advanced_master.py       # Main Gradio Web Application
├── advanced_dataset_loader.py    # Core colorization pipeline & model loader
├── cnn_model.py                 # 4-Stage Convolutional Encoder-Decoder
├── advanced_ddcolor_model.py    # DDColor Dual-Decoder Architecture
├── unet_model.py                # U-Net Architecture with Skip Connections
├── models/                      # Lightweight model checkpoints (<50MB)
│   ├── cnn_best.pth
│   └── advanced_ddcolor_best.pth
├── templates/                   # Web templates (Flask fallback)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 📜 Citation & Acknowledgements

- **DDColor**: Kang et al., *“DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders”*, ICCV 2023.
- **COCO Dataset**: Lin et al., *“Microsoft COCO: Common Objects in Context”*, ECCV 2014.
- **Guided Filter**: He et al., *“Guided Image Filtering”*, IEEE TPAMI 2013.

---

## 👨‍💻 Author

**Nandan K N**  
Department of Information Science & Engineering  
APS College of Engineering, Bengaluru
