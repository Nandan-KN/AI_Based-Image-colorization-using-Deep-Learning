import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity


TEST_MANIFEST = "advanced_dataset/splits/test.txt"
IMAGE_DIR = "advanced_dataset/images"
DDCOLOR_DIR = "advanced_dataset/teacher_outputs/test"

OUTPUT_REPORT = "advanced_outputs/ddcolor_test_metrics.txt"

IMAGE_SIZE = 256


def lab_to_rgb(L, AB):
    """Reconstruct RGB using the same Lab convention as our evaluation."""

    L_channel = np.clip(
        L * 255.0, 0, 255
    ).astype(np.uint8)

    A_channel = np.clip(
        AB[0] * 128.0 + 128.0, 0, 255
    ).astype(np.uint8)

    B_channel = np.clip(
        AB[1] * 128.0 + 128.0, 0, 255
    ).astype(np.uint8)

    lab = np.stack(
        [L_channel, A_channel, B_channel],
        axis=-1
    )

    return cv2.cvtColor(
        lab, cv2.COLOR_LAB2RGB
    )


def main():

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
    print("DDCOLOR TEST METRICS")
    print("=" * 60)
    print("Test images:", len(image_names))

    mse_values = []
    mae_values = []
    psnr_values = []
    ssim_values = []
    missing = []

    for index, name in enumerate(image_names, start=1):

        original_path = os.path.join(
            IMAGE_DIR, name
        )

        ddcolor_path = os.path.join(
            DDCOLOR_DIR, name
        )

        original_bgr = cv2.imread(
            original_path
        )

        ddcolor_bgr = cv2.imread(
            ddcolor_path
        )

        if original_bgr is None or ddcolor_bgr is None:
            missing.append(name)
            continue

        # Match the 256 x 256 preprocessing used
        # by our advanced model.
        original_rgb = cv2.cvtColor(
            original_bgr,
            cv2.COLOR_BGR2RGB
        )

        original_rgb = cv2.resize(
            original_rgb,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        # Reconstruct the evaluation target using Lab,
        # matching our advanced model evaluation.
        lab = cv2.cvtColor(
            original_rgb,
            cv2.COLOR_RGB2LAB
        )

        L = lab[:, :, 0].astype(
            np.float32
        ) / 255.0

        A = (
            lab[:, :, 1].astype(np.float32) - 128.0
        ) / 128.0

        B = (
            lab[:, :, 2].astype(np.float32) - 128.0
        ) / 128.0

        target_AB = np.stack(
            [A, B],
            axis=0
        )

        target_rgb = lab_to_rgb(
            L,
            target_AB
        )

        # Read and resize DDColor output.
        ddcolor_rgb = cv2.cvtColor(
            ddcolor_bgr,
            cv2.COLOR_BGR2RGB
        )

        ddcolor_rgb = cv2.resize(
            ddcolor_rgb,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        target = (
            target_rgb.astype(np.float32) / 255.0
        )

        prediction = (
            ddcolor_rgb.astype(np.float32) / 255.0
        )

        difference = prediction - target

        mse = float(
            np.mean(difference ** 2)
        )

        mae = float(
            np.mean(np.abs(difference))
        )

        psnr = (
            float("inf")
            if mse == 0
            else 10.0 * np.log10(1.0 / mse)
        )

        ssim = structural_similarity(
            target,
            prediction,
            channel_axis=-1,
            data_range=1.0
        )

        mse_values.append(mse)
        mae_values.append(mae)
        psnr_values.append(psnr)
        ssim_values.append(float(ssim))

        if index % 100 == 0:
            print(
                f"Processed {index}/{len(image_names)}"
            )

    if missing:
        print("\nMissing/unreadable pairs:", len(missing))
        print("First missing files:", missing[:10])

    if not mse_values:
        raise RuntimeError(
            "No valid test image pairs were evaluated."
        )

    results = {
        "Test images evaluated": len(mse_values),
        "MSE": float(np.mean(mse_values)),
        "MAE": float(np.mean(mae_values)),
        "PSNR (dB)": float(np.mean(psnr_values)),
        "SSIM": float(np.mean(ssim_values)),
        "Missing/unreadable pairs": len(missing),
    }

    print("\n" + "=" * 60)
    print("FINAL DDCOLOR TEST RESULTS")
    print("=" * 60)

    for key, value in results.items():

        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    os.makedirs(
        os.path.dirname(OUTPUT_REPORT),
        exist_ok=True
    )

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    print("\nReport saved to:", OUTPUT_REPORT)
    print("DDColor evaluation completed.")


if __name__ == "__main__":
    main()