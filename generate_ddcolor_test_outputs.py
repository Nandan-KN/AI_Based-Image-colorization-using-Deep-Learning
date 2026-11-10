import os
import time
import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


# Test images only — do not train on these images.
TEST_MANIFEST = "advanced_dataset/splits/test.txt"

IMAGE_DIR = "advanced_dataset/images"

OUTPUT_DIR = "advanced_dataset/teacher_outputs/test"

MODEL_ID = "damo/cv_ddcolor_image-colorization"


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(
        TEST_MANIFEST,
        "r",
        encoding="utf-8"
    ) as f:
        image_names = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print("=" * 60)
    print("DDCOLOR TEST OUTPUT GENERATION")
    print("=" * 60)
    print("Test images:", len(image_names))
    print("Output folder:", OUTPUT_DIR)

    print("\nLoading pretrained DDColor...")

    colorizer = pipeline(
        Tasks.image_colorization,
        model=MODEL_ID
    )

    saved = 0
    failed = []
    start_time = time.time()

    for index, name in enumerate(image_names, start=1):

        input_path = os.path.join(IMAGE_DIR, name)
        output_path = os.path.join(OUTPUT_DIR, name)

        if not os.path.isfile(input_path):
            failed.append((name, "Original image missing"))
            continue

        # Skip successfully generated outputs when resuming.
        if (
            os.path.isfile(output_path)
            and os.path.getsize(output_path) > 0
        ):
            saved += 1
            continue

        try:
            result = colorizer(input_path)
            output_image = result[OutputKeys.OUTPUT_IMG]

            if output_image is None:
                raise RuntimeError("DDColor returned no image")

            if not cv2.imwrite(output_path, output_image):
                raise RuntimeError("Could not save output")

            saved += 1

        except Exception as exc:
            failed.append((name, str(exc)))

        if index % 50 == 0 or index == len(image_names):
            print(
                f"Processed {index}/{len(image_names)} | "
                f"Saved: {saved} | Failed: {len(failed)}"
            )

    print("\n" + "=" * 60)
    print("DDCOLOR TEST GENERATION COMPLETED")
    print("=" * 60)
    print("Test images:", len(image_names))
    print("Saved or already available:", saved)
    print("Failed:", len(failed))
    print(f"Elapsed time: {time.time() - start_time:.1f} seconds")
    print("Output folder:", OUTPUT_DIR)

    if failed:
        print("\nFirst failed files:")
        for name, reason in failed[:10]:
            print(name, "-", reason)


if __name__ == "__main__":
    main()
    