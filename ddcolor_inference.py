import os
import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


MODEL_ID = "damo/cv_ddcolor_image-colorization"


def load_model():
    print("Loading DDColor model...")

    colorizer = pipeline(
        Tasks.image_colorization,
        model=MODEL_ID
    )

    print("DDColor model loaded successfully.")

    return colorizer


def colorize_image(colorizer, input_path, output_path):
    print(f"Input : {input_path}")
    print("Colorizing image...")

    result = colorizer(input_path)

    output_image = result[OutputKeys.OUTPUT_IMG]

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cv2.imwrite(
        output_path,
        output_image
    )

    print(f"Output: {output_path}")


if __name__ == "__main__":

    input_path = "uploads/image1.jpg"

    output_path = (
        "outputs/new_image_tests/"
        "ddcolor_test.jpg"
    )

    colorizer = load_model()

    colorize_image(
        colorizer,
        input_path,
        output_path
    )

    print("DDColor inference completed.")