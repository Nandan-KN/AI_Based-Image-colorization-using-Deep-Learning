import os
import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


INPUT_PATH = "advanced_dataset/images/000000378491.jpg"

ADVANCED_COMPARISON = (
    "advanced_outputs/test_results/comparison_001.jpg"
)

OUTPUT_DIR = "advanced_outputs/ddcolor_comparison"

DDCOLOR_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "ddcolor_000000378491.jpg"
)

COMBINED_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "comparison_with_ddcolor.jpg"
)


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(INPUT_PATH)

    if not os.path.isfile(ADVANCED_COMPARISON):
        raise FileNotFoundError(ADVANCED_COMPARISON)

    print("Loading pretrained DDColor...")

    colorizer = pipeline(
        Tasks.image_colorization,
        model="damo/cv_ddcolor_image-colorization"
    )

    print("Running DDColor on the same test image...")

    result = colorizer(INPUT_PATH)
    ddcolor_image = result[OutputKeys.OUTPUT_IMG]

    if ddcolor_image is None:
        raise RuntimeError("DDColor returned no output image.")

    cv2.imwrite(
        DDCOLOR_OUTPUT,
        ddcolor_image
    )

    # Load the existing grayscale / advanced / ground-truth
    # comparison created during advanced-model evaluation.
    existing = cv2.imread(
        ADVANCED_COMPARISON
    )

    if existing is None:
        raise RuntimeError(
            f"Could not read {ADVANCED_COMPARISON}"
        )

    height, width = existing.shape[:2]

    panel_width = width // 3
    panel_height = height - 32

    grayscale = existing[
        32:, 0:panel_width
    ]

    advanced = existing[
        32:, panel_width:2 * panel_width
    ]

    ground_truth = existing[
        32:, 2 * panel_width:3 * panel_width
    ]

    ddcolor_image = cv2.resize(
        ddcolor_image,
        (panel_width, panel_height),
        interpolation=cv2.INTER_AREA
    )

    panels = [
        ("Grayscale input", grayscale),
        ("Our advanced model", advanced),
        ("DDColor", ddcolor_image),
        ("Ground truth", ground_truth)
    ]

    label_height = 32

    comparison = cv2.cvtColor(
        cv2.resize(
            cv2.cvtColor(
                cv2.imread(INPUT_PATH),
                cv2.COLOR_BGR2RGB
            ),
            (panel_width, panel_height)
        ),
        cv2.COLOR_RGB2BGR
    )

    canvas = __import__("numpy").zeros(
        (
            panel_height + label_height,
            panel_width * len(panels),
            3
        ),
        dtype="uint8"
    )

    for index, (label, panel) in enumerate(panels):

        x = index * panel_width

        canvas[
            label_height:,
            x:x + panel_width
        ] = panel

        cv2.putText(
            canvas,
            label,
            (x + 5, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(
        COMBINED_OUTPUT,
        canvas
    )

    print()
    print("DDColor output:", DDCOLOR_OUTPUT)
    print("Four-panel comparison:", COMBINED_OUTPUT)
    print("DDColor comparison completed.")


if __name__ == "__main__":
    main()