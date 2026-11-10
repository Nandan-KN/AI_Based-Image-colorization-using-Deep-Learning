import os
import time

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

TRAIN_MANIFEST = "advanced_dataset/splits/train.txt"
IMAGE_DIR = "advanced_dataset/images"
OUTPUT_DIR = "advanced_dataset/teacher_outputs/train"

MODEL_ID = "damo/cv_ddcolor_image-colorization"
MAX_IMAGES = 18000


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isfile(TRAIN_MANIFEST):
        raise FileNotFoundError(TRAIN_MANIFEST)

    with open(TRAIN_MANIFEST, "r", encoding="utf-8") as f:
        image_names = [line.strip() for line in f if line.strip()]

    image_names = image_names[:MAX_IMAGES]

    print("Training images selected:", len(image_names))
    print("Teacher output folder:", OUTPUT_DIR)
    print("Loading pretrained DDColor...")

    colorizer = pipeline(
        Tasks.image_colorization,
        model=MODEL_ID
    )

    succeeded = 0
    failed = []
    start = time.time()

    for index, name in enumerate(image_names, start=1):
        input_path = os.path.join(IMAGE_DIR, name)
        output_path = os.path.join(OUTPUT_DIR, name)

        if not os.path.isfile(input_path):
            failed.append((name, "Input image not found"))
            continue

        # Allow interrupted runs to resume.
        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            succeeded += 1
            continue

        try:
            result = colorizer(input_path)
            output_image = result[OutputKeys.OUTPUT_IMG]

            if output_image is None:
                raise RuntimeError("DDColor returned no image")

            # Match the output-saving convention used by our
            # existing DDColor inference script.
            import cv2

            if not cv2.imwrite(output_path, output_image):
                raise RuntimeError("Could not save output image")

            succeeded += 1

        except Exception as exc:
            failed.append((name, str(exc)))

        if index % 10 == 0 or index == len(image_names):
            print(
                f"Processed {index}/{len(image_names)} | "
                f"Saved {succeeded} | Failed {len(failed)}"
            )

    print("\nTeacher generation finished.")
    print("Successful outputs:", succeeded)
    print("Failures:", len(failed))
    print(f"Elapsed time: {time.time() - start:.1f} seconds")

    if failed:
        print("First failures:")
        for name, reason in failed[:5]:
            print(name, "-", reason)


if __name__ == "__main__":
    main()