"""
streamlit_app.py - 24/7 Cloud Deployment Entrypoint for Streamlit Community Cloud
"""

import time
from io import BytesIO
from PIL import Image
import numpy as np
import streamlit as st
import cv2

from advanced_dataset_loader import colorize_image, preload_all_models

st.set_page_config(
    page_title="AI Image Colorization",
    page_icon="🎨",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">AI Image Colorization</h1>', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    preload_all_models()
    return True

load_models()

MODEL_CHOICES = {
    "CNN Model": "models/cnn_best.pth",
    "DDColor": "models/advanced_ddcolor_best.pth",
    "ResNet-18 Chroma": "models/resnet18_chroma_epoch1.pth",
}

# Sidebar Controls
st.sidebar.header("⚙️ Settings & Controls")
model_choice = st.sidebar.radio(
    "Select AI Model Engine",
    options=["CNN Model", "DDColor", "ResNet-18 Chroma"],
    index=0
)

with st.sidebar.expander("🎨 Advanced Color Tuning", expanded=False):
    mood_choice = st.selectbox(
        "Color Harmony Preset",
        options=["Natural Balanced", "Vivid Rich", "Warm Golden Hour", "Cool Cinematic"],
        index=0
    )
    vibrancy = st.slider("Color Vibrancy (Chroma Gamut Expansion)", 1.0, 2.5, 1.85, 0.05)
    vibrance_gain = st.slider("Photographic Vibrance", 0.0, 1.0, 0.45, 0.05)
    edge_lock = st.checkbox("Edge-Aware Anti-Bleed Lock", value=True)

# Main interface
uploaded_file = st.file_uploader(
    "Choose a grayscale or historical photograph...",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    input_pil = Image.open(uploaded_file).convert("RGB")
    input_np = np.array(input_pil)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Grayscale Input")
        st.image(input_pil, use_container_width=True)

    with col2:
        st.subheader("Enhanced Colorized Result")
        with st.spinner("Colorizing image with deep learning..."):
            t0 = time.perf_counter()
            model_path = MODEL_CHOICES.get(model_choice, "models/cnn_best.pth")
            
            output_rgb = colorize_image(
                input_np,
                model_path=model_path,
                chroma_boost=float(vibrancy),
                vibrance=float(vibrance_gain),
                saturation_boost=1.0,
                mood=mood_choice,
                enable_edge_lock=bool(edge_lock),
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            h, w = output_rgb.shape[:2]

        output_pil = Image.fromarray(output_rgb)
        st.image(output_pil, use_container_width=True)
        st.success(f"✨ **Colorization Complete** | ⚡ **{elapsed_ms:.1f} ms** | 📐 **{w} × {h} px** | 🧠 Engine: **{model_choice}**")

        # Download button
        buf = BytesIO()
        output_pil.save(buf, format="JPEG", quality=95)
        byte_im = buf.getvalue()
        st.download_button(
            label="📥 Download Colorized Image",
            data=byte_im,
            file_name="colorized_output.jpg",
            mime="image/jpeg"
        )
else:
    st.info("👆 Please upload an image above to begin colorization.")
