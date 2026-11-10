"""
app_resnet_colorizer.py
=======================
Upgraded Multi-Engine AI Image Colorization Web Application
-----------------------------------------------------------
Features:
1. Multi-Model Engine Switcher:
   - ResNet-18 Chroma (Champion)
   - Teacher DDColor Transformer (State-of-the-art Deep Semantic)
   - Custom 5-Stage Deep U-Net (Pixel-Perceptual Loss)
2. Edge-Aware Structural Locking (Joint Guided Filter Anti-Bleeding)
3. Artistic Mood / Color Grading Presets (Natural, Vibrant, Sunset Warmth, Cool Cinematic, Vintage Film)
4. Interactive Draggable Split-Screen Before/After Slider
5. Real-Time Telemetry & Image Analytics (Resolution, Latency, Hardware Device)
6. Sample Photo Quick-Launcher
7. Programmatic REST API (/api/colorize)
"""

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import torch
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    jsonify,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename

from advanced_resnet_colorizer import ResNet18Colorizer
from advanced_ddcolor_model import AdvancedDDColorInspired
from unet_model import ColorizationUNet
from advanced_refiner_models import edge_aware_structural_lock


# ==============================================================================
# Configuration & Directory Paths
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads_resnet"
OUTPUT_FOLDER = BASE_DIR / "advanced_outputs" / "resnet_web_results"
SAMPLES_FOLDER = BASE_DIR / "uploads"

IMAGE_SIZE = 256
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Flask Application
# ==============================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


# ==============================================================================
# Device & Multi-Model Registry
# ==============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU Host Processor"

print(f"[*] Initializing Multi-Engine Colorization Framework on {device} ({device_name})")

models_cache = {
    "resnet18": None,
    "teacher_ddcolor": None,
    "unet": None,
}


def get_model(engine_name: str):
    """Lazily load and cache models into memory."""
    engine_name = engine_name.lower().strip()

    if engine_name in models_cache and models_cache[engine_name] is not None:
        return models_cache[engine_name]

    if engine_name == "resnet18":
        ckpt = BASE_DIR / "models" / "resnet18_chroma_epoch1.pth"
        print(f"[*] Loading ResNet-18 Chroma from {ckpt.name}...")
        model = ResNet18Colorizer(pretrained=False).to(device)
        state = torch.load(str(ckpt), map_location=device)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        model.eval()
        models_cache["resnet18"] = model
        return model

    elif engine_name in ["teacher", "teacher_ddcolor", "ddcolor"]:
        ckpt = BASE_DIR / "models" / "advanced_ddcolor_teacher_w15_epoch1.pth"
        print(f"[*] Loading Teacher DDColor Transformer from {ckpt.name}...")
        model = AdvancedDDColorInspired(num_queries=16).to(device)
        state = torch.load(str(ckpt), map_location=device)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        model.eval()
        models_cache["teacher_ddcolor"] = model
        return model

    elif engine_name == "unet":
        ckpt = BASE_DIR / "models" / "unet_colorizer_best.pth"
        print(f"[*] Loading 5-Stage U-Net from {ckpt.name}...")
        model = ColorizationUNet().to(device)
        state = torch.load(str(ckpt), map_location=device)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        model.eval()
        models_cache["unet"] = model
        return model

    else:
        # Fallback to default
        return get_model("resnet18")


# Preload primary champion model
try:
    get_model("resnet18")
except Exception as e:
    print(f"[!] Warning: Failed preloading ResNet18: {e}")


# ==============================================================================
# Helper Functions
# ==============================================================================

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def apply_color_mood(
    image_bgr: np.ndarray,
    mood: str,
    saturation_boost: float = 1.15
) -> np.ndarray:
    """Applies artistic tone grading and saturation calibration in HSV/LAB space."""
    mood = mood.lower().strip()

    # Convert to RGB & HSV
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Base saturation boost
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0.0, 255.0)

    if mood == "vibrant":
        # Extra chroma punch
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0.0, 255.0)
        rgb_graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    elif mood == "warm_sunset":
        # Boost warm reds and golden yellows
        lab = cv2.cvtColor(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + 6.0, 0.0, 255.0)   # +a* (red warmth)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + 10.0, 0.0, 255.0)  # +b* (golden yellow)
        rgb_graded = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    elif mood == "cool_cinematic":
        # Cinematic teal & cyan shadow shift
        lab = cv2.cvtColor(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 1] = np.clip(lab[:, :, 1] - 4.0, 0.0, 255.0)   # cooler greens
        lab[:, :, 2] = np.clip(lab[:, :, 2] - 8.0, 0.0, 255.0)   # deep cyans/blues
        rgb_graded = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    elif mood == "vintage_retro":
        # Kodachrome 1970s soft contrast and sepia warmth
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.85, 0.0, 255.0)
        lab = cv2.cvtColor(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + 12.0, 0.0, 255.0)  # vintage amber tint
        rgb_graded = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    else:
        # Default Natural
        rgb_graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return cv2.cvtColor(rgb_graded, cv2.COLOR_RGB2BGR)


def colorize_pipeline(
    input_path: Path,
    output_path: Path,
    engine: str = "resnet18",
    mood: str = "natural",
    enable_edge_lock: bool = True,
    saturation: float = 1.15
) -> Dict[str, Any]:
    """Complete image colorization pipeline with multi-engine support and structural locking."""
    t_start = time.perf_counter()

    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode the uploaded image file.")

    orig_h, orig_w = image_bgr.shape[:2]

    # 1. Color Conversion (Strict BGR -> RGB -> LAB to preserve channel alignment)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    native_L = lab[:, :, 0]

    # 2. Resize L channel for neural processing (256x256)
    L_small = cv2.resize(native_L, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    L_normalized = L_small.astype(np.float32) / 255.0
    L_tensor = torch.from_numpy(L_normalized).unsqueeze(0).unsqueeze(0).to(device)

    # 3. Model Inference
    model = get_model(engine)
    with torch.inference_mode():
        predicted_AB = model(L_tensor)[0].float().cpu().numpy()

    # 4. Upsample chrominance channels to native resolution
    pred_a = cv2.resize(predicted_AB[0], (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    pred_b = cv2.resize(predicted_AB[1], (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # 5. Reverse training normalization: center around 128
    a_uint8 = np.clip(pred_a * 128.0 + 128.0, 0, 255).astype(np.uint8)
    b_uint8 = np.clip(pred_b * 128.0 + 128.0, 0, 255).astype(np.uint8)

    # 6. Merge with native high-res L channel
    colorized_lab = cv2.merge([native_L, a_uint8, b_uint8])
    colorized_rgb = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2RGB)
    colorized_bgr = cv2.cvtColor(colorized_rgb, cv2.COLOR_RGB2BGR)

    # 7. Apply Artistic Color Mood Grading
    colorized_bgr = apply_color_mood(colorized_bgr, mood=mood, saturation_boost=saturation)

    # 8. Edge-Aware Structural Locking (Anti-Bleeding via Joint Guided Filter)
    if enable_edge_lock:
        colorized_bgr, _ = edge_aware_structural_lock(
            color_bgr=colorized_bgr,
            gray_guide=native_L,
            radius=6,
            eps=0.005
        )

    # 9. Save final output
    if not cv2.imwrite(str(output_path), colorized_bgr):
        raise RuntimeError("Failed to write colorized image to disk.")

    t_elapsed = time.perf_counter() - t_start
    latency_ms = round(t_elapsed * 1000, 1)

    engine_titles = {
        "resnet18": "ResNet-18 Chroma (Champion)",
        "teacher": "Teacher DDColor Transformer",
        "teacher_ddcolor": "Teacher DDColor Transformer",
        "unet": "Custom 5-Stage U-Net",
    }

    mood_titles = {
        "natural": "Natural & Balanced",
        "vibrant": "Vibrant Pop",
        "warm_sunset": "Warm Golden Sunset",
        "cool_cinematic": "Cool Cinematic Teal",
        "vintage_retro": "Vintage 1970s Retro",
    }

    return {
        "width": orig_w,
        "height": orig_h,
        "megapixels": round((orig_w * orig_h) / 1_000_000, 2),
        "latency_ms": latency_ms,
        "engine_name": engine_titles.get(engine, "ResNet-18 Chroma"),
        "engine_key": engine,
        "mood_name": mood_titles.get(mood, "Natural"),
        "mood_key": mood,
        "edge_locked": enable_edge_lock,
        "device_name": device_name,
    }


# ==============================================================================
# Routes & Controllers
# ==============================================================================

@app.route("/")
def home():
    """Renders modern interactive dashboard."""
    # List sample images for quick click-to-test
    sample_files = []
    if SAMPLES_FOLDER.is_dir():
        for f in SAMPLES_FOLDER.glob("*.*"):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"] and f.name.startswith("image"):
                sample_files.append(f.name)
    sample_files = sorted(sample_files)[:4]

    return render_template(
        "resnet_index.html",
        sample_files=sample_files,
        device_name=device_name,
    )


@app.route("/colorize", methods=["POST"])
def colorize():
    """Main colorization request handler."""
    engine = request.form.get("engine", "resnet18")
    mood = request.form.get("mood", "natural")
    enable_edge_lock = request.form.get("edge_lock") in ["true", "on", "1"]
    saturation = float(request.form.get("saturation", 1.15))

    # Check for file upload or sample selection
    sample_name = request.form.get("sample_image", "").strip()

    if sample_name and (SAMPLES_FOLDER / sample_name).is_file():
        sample_path = SAMPLES_FOLDER / sample_name
        token = uuid.uuid4().hex
        input_filename = f"{token}_{sample_name}"
        input_path = UPLOAD_FOLDER / input_filename
        # Copy sample into upload folder
        with open(sample_path, "rb") as src, open(input_path, "wb") as dst:
            dst.write(src.read())
    else:
        if "image" not in request.files:
            return render_template("resnet_index.html", error="Please choose an image file."), 400
        
        uploaded_file = request.files["image"]
        if uploaded_file.filename == "":
            return render_template("resnet_index.html", error="No image was selected."), 400

        if not allowed_file(uploaded_file.filename):
            return render_template("resnet_index.html", error="Supported formats: JPG, JPEG, PNG, WEBP."), 400

        safe_name = secure_filename(uploaded_file.filename)
        token = uuid.uuid4().hex
        input_filename = f"{token}_{safe_name}"
        input_path = UPLOAD_FOLDER / input_filename
        uploaded_file.save(str(input_path))

    output_filename = f"colorized_{token}.png"
    output_path = OUTPUT_FOLDER / output_filename

    try:
        telemetry = colorize_pipeline(
            input_path=input_path,
            output_path=output_path,
            engine=engine,
            mood=mood,
            enable_edge_lock=enable_edge_lock,
            saturation=saturation,
        )
    except Exception as exc:
        print(f"[!] Colorization error: {exc}")
        return render_template(
            "resnet_index.html",
            error=f"Processing failed: {str(exc)}",
        ), 500

    return render_template(
        "resnet_result.html",
        input_image=input_filename,
        output_image=output_filename,
        meta=telemetry,
    )


@app.route("/api/colorize", methods=["POST"])
def api_colorize():
    """REST API endpoint for headless and programmatic colorization."""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image field in multipart request"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid image format"}), 400

    engine = request.form.get("engine", "resnet18")
    mood = request.form.get("mood", "natural")
    enable_edge_lock = request.form.get("edge_lock", "true").lower() in ["true", "1"]
    saturation = float(request.form.get("saturation", 1.15))

    token = uuid.uuid4().hex
    input_filename = f"api_{token}_{secure_filename(file.filename)}"
    output_filename = f"colorized_{token}.png"
    input_path = UPLOAD_FOLDER / input_filename
    output_path = OUTPUT_FOLDER / output_filename

    file.save(str(input_path))

    try:
        telemetry = colorize_pipeline(
            input_path=input_path,
            output_path=output_path,
            engine=engine,
            mood=mood,
            enable_edge_lock=enable_edge_lock,
            saturation=saturation,
        )
        return jsonify({
            "success": True,
            "input_url": f"/uploads/{input_filename}",
            "colorized_url": f"/results/{output_filename}",
            "download_url": f"/download/{output_filename}",
            "telemetry": telemetry,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.route("/results/<path:filename>")
def result_file(filename):
    return send_from_directory(str(OUTPUT_FOLDER), filename)


@app.route("/samples/<path:filename>")
def sample_file(filename):
    return send_from_directory(str(SAMPLES_FOLDER), filename)


@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(
        str(OUTPUT_FOLDER),
        filename,
        as_attachment=True,
        download_name="colorized_image.png",
    )


# ==============================================================================
# Run Application Server
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])

    print(f"[*] Starting Colorization Web Server on http://127.0.0.1:{port}")
    try:
        app.run(host="127.0.0.1", port=port, debug=False)
    except OSError:
        fallback = 5004
        print(f"[!] Port {port} in use, launching on fallback port http://127.0.0.1:{fallback}")
        app.run(host="127.0.0.1", port=fallback, debug=False)
