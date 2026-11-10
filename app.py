import os
from flask import Flask, request, send_file
from inference import colorize_image

app = Flask(__name__)

@app.route("/")
def home():
    return "AI Image Colorization API is running."

@app.route("/colorize", methods=["POST"])
def colorize():
    if "image" not in request.files:
        return {"error": "No image uploaded"}, 400

    file = request.files["image"]
    if file.filename == "":
        return {"error": "No selected file"}, 400

    input_path = os.path.join("tmp", "input.jpg")
    output_path = os.path.join("tmp", "output.jpg")

    os.makedirs("tmp", exist_ok=True)

    file.save(input_path)
    colorize_image(input_path, output_path)

    return send_file(output_path, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)