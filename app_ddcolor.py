import os

from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename

from ddcolor_inference import load_model, colorize_image


app = Flask(__name__)


UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs/ddcolor_web_results"

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


print("Loading DDColor model...")
colorizer = load_model()
print("DDColor model ready.")


def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():

    return render_template(
        "ddcolor_index.html"
    )


@app.route("/colorize", methods=["POST"])
def colorize():

    if "image" not in request.files:

        return "No image uploaded."


    file = request.files["image"]


    if file.filename == "":

        return "No image selected."


    if not allowed_file(file.filename):

        return "Unsupported image format."


    filename = secure_filename(
        file.filename
    )


    input_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    file.save(input_path)


    output_filename = (
        "ddcolor_"
        + filename
    )


    output_path = os.path.join(
        app.config["OUTPUT_FOLDER"],
        output_filename
    )


    colorize_image(
        colorizer,
        input_path,
        output_path
    )


    return render_template(
        "ddcolor_result.html",
        input_image=filename,
        output_image=output_filename
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


@app.route("/results/<filename>")
def result_file(filename):

    return send_from_directory(
        app.config["OUTPUT_FOLDER"],
        filename
    )


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
    
    try:
        colorize_image(colorizer, input_path, output_path)
    except Exception as error:
        print("Colorization error:", error)
        return jsonify({"error": "Colorization failed"}), 500
    
    return jsonify({
        "success": True,
        "input_image": filename,
        "output_image": output_filename,
        "output_url": f"/results/{output_filename}"
    })


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5003,
        debug=False
    )