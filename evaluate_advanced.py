import os
import time

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

from advanced_ddcolor_model import AdvancedDDColorInspired
from advanced_split_loader import create_advanced_loaders


CHECKPOINT_PATH = "models/advanced_ddcolor_best.pth"
OUTPUT_DIR = "advanced_outputs/test_results"

BATCH_SIZE = 2
IMAGE_SIZE = 256
NUM_VISUAL_SAMPLES = 10


def lab_to_rgb(L, AB):
    """Convert normalized L and A/B channels to an RGB image."""

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


def save_comparison(gray, prediction, target, index):
    """Save grayscale input, prediction, and target side by side."""

    height, width = gray.shape[:2]
    labels = [
        "Grayscale input",
        "Advanced output",
        "Ground truth"
    ]

    panels = [gray, prediction, target]

    canvas = np.zeros(
        (height + 32, width * 3, 3),
        dtype=np.uint8
    )

    for i, (panel, label) in enumerate(
        zip(panels, labels)
    ):
        x = i * width

        canvas[32:, x:x + width] = cv2.cvtColor(
            panel, cv2.COLOR_RGB2BGR
        )

        cv2.putText(
            canvas,
            label,
            (x + 6, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"comparison_{index:03d}.jpg"
    )

    cv2.imwrite(output_path, canvas)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("ADVANCED MODEL TEST EVALUATION")
    print("=" * 65)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load the best checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model = AdvancedDDColorInspired(
        num_queries=checkpoint.get("num_queries", 16)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Best validation loss:",
        checkpoint["best_validation_loss"]
    )

    # Load the fixed test split only.
    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader
    ) = create_advanced_loaders(
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    print("Unseen test images:", len(test_dataset))

    squared_errors = []
    absolute_errors = []
    psnr_scores = []
    ssim_scores = []

    processed = 0
    start_time = time.time()

    with torch.inference_mode():
        for L_batch, target_AB_batch in test_loader:

            L_batch = L_batch.to(device)

            predicted_AB_batch = model(L_batch)

            L_numpy = L_batch.cpu().numpy()
            target_numpy = target_AB_batch.numpy()
            prediction_numpy = predicted_AB_batch.cpu().numpy()

            for i in range(L_batch.size(0)):

                L = L_numpy[i, 0]
                target_AB = target_numpy[i]
                predicted_AB = prediction_numpy[i]

                target_rgb = lab_to_rgb(L, target_AB)
                predicted_rgb = lab_to_rgb(L, predicted_AB)

                # Normalize RGB to [0, 1].
                target_float = target_rgb.astype(
                    np.float32
                ) / 255.0

                prediction_float = predicted_rgb.astype(
                    np.float32
                ) / 255.0

                difference = (
                    prediction_float - target_float
                )

                mse = float(np.mean(difference ** 2))
                mae = float(np.mean(np.abs(difference)))

                psnr = (
                    float("inf")
                    if mse == 0
                    else 10.0 * np.log10(1.0 / mse)
                )

                ssim = structural_similarity(
                    target_float,
                    prediction_float,
                    channel_axis=-1,
                    data_range=1.0
                )

                squared_errors.append(mse)
                absolute_errors.append(mae)
                psnr_scores.append(psnr)
                ssim_scores.append(float(ssim))

                processed += 1

                # Save only the first ten fixed test samples.
                if processed <= NUM_VISUAL_SAMPLES:

                    gray_rgb = np.repeat(
                        np.clip(
                            L * 255.0, 0, 255
                        ).astype(np.uint8)[..., None],
                        3,
                        axis=2
                    )

                    save_comparison(
                        gray_rgb,
                        predicted_rgb,
                        target_rgb,
                        processed
                    )

            if processed % 100 == 0:
                print(
                    f"Evaluated {processed}/"
                    f"{len(test_dataset)} images"
                )

    print()
    print("=" * 65)
    print("FINAL TEST METRICS")
    print("=" * 65)

    print("Test images evaluated:", processed)
    print(f"MSE  : {np.mean(squared_errors):.6f}")
    print(f"MAE  : {np.mean(absolute_errors):.6f}")
    print(f"PSNR : {np.mean(psnr_scores):.4f} dB")
    print(f"SSIM : {np.mean(ssim_scores):.4f}")

    print()
    print("Visual comparisons saved to:", OUTPUT_DIR)
    print("Evaluation time:", f"{time.time() - start_time:.1f} seconds")

    # Save a text report.
    report_path = os.path.join(
        OUTPUT_DIR,
        "test_metrics.txt"
    )

    with open(report_path, "w", encoding="utf-8") as report:
        report.write(f"Checkpoint epoch: {checkpoint['epoch']}\n")
        report.write(f"Test images: {processed}\n")
        report.write(f"MSE: {np.mean(squared_errors):.6f}\n")
        report.write(f"MAE: {np.mean(absolute_errors):.6f}\n")
        report.write(f"PSNR: {np.mean(psnr_scores):.4f} dB\n")
        report.write(f"SSIM: {np.mean(ssim_scores):.4f}\n")

    print("Metrics report saved to:", report_path)
    print("ADVANCED MODEL EVALUATION COMPLETED")


if __name__ == "__main__":
    main()