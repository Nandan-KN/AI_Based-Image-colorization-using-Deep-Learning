from pathlib import Path
import cv2
import numpy as np

from ddcolor_inference import load_model, colorize_image


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path.cwd()

TEST_MANIFEST = (
    PROJECT_DIR / "advanced_dataset" / "splits" / "test.txt"
)

IMAGE_DIR = PROJECT_DIR / "advanced_dataset" / "images"

OUTPUT_DIR = (
    PROJECT_DIR
    / "advanced_outputs"
    / "checkpoint_comparison"
)

EXISTING_COMPARISON = OUTPUT_DIR / "sample_001.jpg"
DDCOLOR_OUTPUT = OUTPUT_DIR / "ddcolor_sample_001.jpg"
FINAL_COMPARISON = OUTPUT_DIR / "sample_001_all_models.jpg"


def main():

    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(TEST_MANIFEST)

    names = [
        line.strip()
        for line in TEST_MANIFEST.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not names:
        raise RuntimeError("The test manifest is empty.")

    # sample_001.jpg was generated from the first test entry.
    input_image = IMAGE_DIR / names[0]

    if not input_image.exists():
        raise FileNotFoundError(input_image)

    if not EXISTING_COMPARISON.exists():
        raise FileNotFoundError(EXISTING_COMPARISON)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Test image:", input_image)
    print("Loading pretrained DDColor...")

    colorizer = load_model()

    colorize_image(
        colorizer,
        str(input_image),
        str(DDCOLOR_OUTPUT)
    )

    # Load the existing four-panel comparison and DDColor result.
    comparison = cv2.imread(
        str(EXISTING_COMPARISON)
    )

    ddcolor_image = cv2.imread(
        str(DDCOLOR_OUTPUT)
    )

    if comparison is None or ddcolor_image is None:
        raise RuntimeError("Could not read comparison images.")

    # The existing comparison has four square panels.
    panel_size = comparison.shape[0]

    ddcolor_image = cv2.resize(
        ddcolor_image,
        (panel_size, panel_size),
        interpolation=cv2.INTER_AREA
    )

    cv2.putText(
        ddcolor_image,
        "DDColor",
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # Existing image contains input, ground truth, original model,
    # and teacher model. Append the DDColor panel.
    combined = np.concatenate(
        [comparison, ddcolor_image],
        axis=1
    )

    cv2.imwrite(
        str(FINAL_COMPARISON),
        combined
    )

    print()
    print("DDColor inference completed.")
    print("DDColor output:", DDCOLOR_OUTPUT)
    print("Five-panel comparison:", FINAL_COMPARISON)


if __name__ == "__main__":
    main()