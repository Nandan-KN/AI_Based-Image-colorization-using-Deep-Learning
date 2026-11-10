import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

from advanced_ddcolor_model import AdvancedDDColorInspired
from advanced_split_loader import AdvancedSplitDataset


BASE_CHECKPOINT = "models/advanced_ddcolor_best.pth"
PILOT_CHECKPOINT = "models/advanced_teacher_pilot_w15.pth"

TEST_MANIFEST = "advanced_dataset/splits/test.txt"
IMAGE_ROOT = "advanced_dataset/images"

OUTPUT_DIR = "advanced_outputs/teacher_w15_comparison"

BATCH_SIZE = 2
NUM_VISUAL_SAMPLES = 10
IMAGE_SIZE = 256


def lab_to_rgb(L, AB):
    """Convert normalized L and A/B channels into an RGB image."""

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
        lab,
        cv2.COLOR_LAB2RGB
    )


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model = AdvancedDDColorInspired(
        num_queries=checkpoint.get("num_queries", 16)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def calculate_metrics(prediction_rgb, target_rgb):
    prediction = prediction_rgb.astype(np.float32) / 255.0
    target = target_rgb.astype(np.float32) / 255.0

    difference = prediction - target
    mse = float(np.mean(difference ** 2))
    mae = float(np.mean(np.abs(difference)))

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

    return {
        "mse": mse,
        "mae": mae,
        "psnr": float(psnr),
        "ssim": float(ssim)
    }


def save_comparison(
    name,
    gray_rgb,
    base_rgb,
    pilot_rgb,
    target_rgb
):
    panels = [
        ("Grayscale input", gray_rgb),
        ("Original advanced model", base_rgb),
        ("Teacher-guided pilot", pilot_rgb),
        ("Ground truth", target_rgb)
    ]

    header_height = 32
    height, width = gray_rgb.shape[:2]

    canvas = np.zeros(
        (height + header_height, width * 4, 3),
        dtype=np.uint8
    )

    for index, (label, image) in enumerate(panels):
        x = index * width

        canvas[
            header_height:, x:x + width
        ] = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        cv2.putText(
            canvas,
            label,
            (x + 5, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    safe_name = os.path.splitext(
        os.path.basename(name)
    )[0]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}_comparison.jpg"
    )

    cv2.imwrite(
        output_path,
        canvas
    )


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("ORIGINAL MODEL vs TEACHER-GUIDED PILOT")
    print("=" * 65)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load both checkpoints.
    base_checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location=device
    )

    pilot_checkpoint = torch.load(
        PILOT_CHECKPOINT,
        map_location=device
    )

    base_model = load_model(
        BASE_CHECKPOINT,
        device
    )

    pilot_model = load_model(
        PILOT_CHECKPOINT,
        device
    )

    print(
        "Original checkpoint epoch:",
        base_checkpoint.get("epoch", "unknown")
    )

    print(
        "Pilot batches:",
        pilot_checkpoint.get("pilot_batches", "unknown")
    )

    # Use the exact same fixed test manifest.
    test_dataset = AdvancedSplitDataset(
        split_file=TEST_MANIFEST,
        image_root=IMAGE_ROOT,
        image_size=IMAGE_SIZE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

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

    base_metrics = {
        "mse": [],
        "mae": [],
        "psnr": [],
        "ssim": []
    }

    pilot_metrics = {
        "mse": [],
        "mae": [],
        "psnr": [],
        "ssim": []
    }

    processed = 0

    with torch.inference_mode():

        for L_batch, target_AB_batch in test_loader:

            L_device = L_batch.to(
                device,
                non_blocking=True
            )

            base_AB = base_model(
                L_device
            ).cpu().numpy()

            pilot_AB = pilot_model(
                L_device
            ).cpu().numpy()

            L_numpy = L_batch.numpy()
            target_numpy = target_AB_batch.numpy()

            for i in range(L_batch.size(0)):

                L = L_numpy[i, 0]
                target_AB = target_numpy[i]

                target_rgb = lab_to_rgb(
                    L,
                    target_AB
                )

                base_rgb = lab_to_rgb(
                    L,
                    base_AB[i]
                )

                pilot_rgb = lab_to_rgb(
                    L,
                    pilot_AB[i]
                )

                gray = np.clip(
                    L * 255.0, 0, 255
                ).astype(np.uint8)

                gray_rgb = np.repeat(
                    gray[..., None],
                    3,
                    axis=2
                )

                base_result = calculate_metrics(
                    base_rgb,
                    target_rgb
                )

                pilot_result = calculate_metrics(
                    pilot_rgb,
                    target_rgb
                )

                for key in base_metrics:
                    base_metrics[key].append(
                        base_result[key]
                    )
                    pilot_metrics[key].append(
                        pilot_result[key]
                    )

                processed += 1

                if processed <= NUM_VISUAL_SAMPLES:
                    save_comparison(
                        image_names[processed - 1],
                        gray_rgb,
                        base_rgb,
                        pilot_rgb,
                        target_rgb
                    )

            if processed % 200 == 0:
                print(
                    f"Evaluated {processed}/"
                    f"{len(test_dataset)} images"
                )

    report_path = os.path.join(
        OUTPUT_DIR,
        "comparison_metrics.txt"
    )

    print()
    print("=" * 65)
    print("FINAL TEST-SET COMPARISON")
    print("=" * 65)
    print("Images evaluated:", processed)

    report_lines = []

    for model_name, metrics in [
        ("Original advanced model", base_metrics),
        ("Teacher-guided pilot", pilot_metrics)
    ]:
        print()
        print(model_name)

        report_lines.append(model_name)

        for key in metrics:
            mean_value = float(
                np.mean(metrics[key])
            )

            line = f"{key.upper():5s}: {mean_value:.6f}"

            print(line)
            report_lines.append(line)

        report_lines.append("")

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(report_lines))

    print()
    print("Visual comparisons:", OUTPUT_DIR)
    print("Metrics report:", report_path)
    print("Same-test-set comparison completed.")


if __name__ == "__main__":
    main()