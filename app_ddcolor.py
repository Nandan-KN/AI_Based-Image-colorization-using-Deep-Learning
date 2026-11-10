import os
import sys
import time
from pathlib import Path
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np

from ddcolor_inference import load_model, colorize_image
from advanced_refiner_models import edge_aware_structural_lock

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = str(BASE_DIR / "uploads")
OUTPUT_FOLDER = str(BASE_DIR / "outputs" / "ddcolor_web_results")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("[*] Loading DDColor ModelScope Pipeline...")
colorizer = load_model()
print("[*] DDColor pipeline initialized and ready.")


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    # Gather sample images for one-click testing
    samples = []
    if os.path.isdir(UPLOAD_FOLDER):
        for f in os.listdir(UPLOAD_FOLDER):
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and f.startswith("image"):
                samples.append(f)
    samples = sorted(samples)[:4]
    return render_template("ddcolor_index.html", samples=samples)


@app.route("/colorize", methods=["POST"])
def colorize():
    sample_name = request.form.get("sample_image", "").strip()

    if sample_name and os.path.isfile(os.path.join(app.config["UPLOAD_FOLDER"], sample_name)):
        filename = sample_name
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    else:
        if "image" not in request.files:
            return "No image uploaded.", 400

        file = request.files["image"]
        if file.filename == "":
            return "No image selected.", 400

        if not allowed_file(file.filename):
            return "Unsupported image format.", 400

        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(input_path)

    output_filename = "ddcolor_" + filename
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)

    t_start = time.perf_counter()

    # 1. DDColor inference with structural edge lock
    colorize_image(colorizer, input_path, output_path)

    # 2. Extract telemetry metadata
    output_bgr = cv2.imread(output_path)
    if output_bgr is not None:
        h, w = output_bgr.shape[:2]
        megapixels = round((w * h) / 1_000_000, 2)
    else:
        w, h, megapixels = 512, 512, 0.26

    latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

    meta = {
        "width": w,
        "height": h,
        "megapixels": megapixels,
        "latency_ms": latency_ms,
        "architecture": "Dual-Decoder Cross-Attention Transformer",
        "queries": "16 Semantic Color Queries",
        "edge_locked": True,
        "model_id": "damo/cv_ddcolor_image-colorization",
    }

    return render_template(
        "ddcolor_result.html",
        input_image=filename,
        output_image=output_filename,
        meta=meta
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


# ============================================================
# JSON API ENDPOINT
# ============================================================

@app.route("/api/colorize", methods=["POST"])
def api_colorize():
    """JSON API endpoint for programmatic access"""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    output_filename = "ddcolor_" + filename
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)

    file.save(input_path)

    t_start = time.perf_counter()
    try:
        colorize_image(colorizer, input_path, output_path)
        output_bgr = cv2.imread(output_path)
        if output_bgr is not None:
            h, w = output_bgr.shape[:2]
        else:
            w, h = 512, 512
        latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

    except Exception as error:
        print("Colorization error:", error)
        return jsonify({"error": "Colorization failed"}), 500

    return jsonify({
        "success": True,
        "input_image": filename,
        "output_image": output_filename,
        "output_url": f"/results/{output_filename}",
        "latency_ms": latency_ms,
        "width": w,
        "height": h,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])

    print(f"[*] Starting DDColor Web Server on http://127.0.0.1:{port}")
    app.run(
        host="127.0.0.1",
        port=port,
        debug=False
    )
