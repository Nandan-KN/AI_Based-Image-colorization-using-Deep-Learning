import os
import csv

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

from advanced_ddcolor_model import AdvancedDDColorInspired
from advanced_split_loader import AdvancedSplitDataset


BASE_CHECKPOINT = "models/advanced_ddcolor_best.pth"
PILOT_CHECKPOINT = "models/advanced_teacher_pilot.pth"

TEST_MANIFEST = "advanced_dataset/splits/test.txt"
IMAGE_ROOT = "advanced_dataset/images"
DDCOLOR_DIR = "advanced_dataset/teacher_outputs/test"

AUDIT_CSV = "advanced_outputs/evaluation_grayscale_audit.csv"
OUTPUT_DIR = "advanced_outputs/grouped_model_evaluation"

IMAGE_SIZE = 256
BATCH_SIZE = 2


def lab_to_rgb(L, AB):
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
        "MSE": mse,
        "MAE": mae,
        "PSNR": float(psnr),
        "SSIM": float(ssim)
    }


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


def load_near_monochrome_flags():
    flags = {}

    with open(
        AUDIT_CSV,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["split"].strip().lower() != "test":
                continue

            flags[row["filename"]] = (
                row["near_monochrome"].strip().lower() == "true"
            )

    return flags


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("GROUPED MODEL EVALUATION")
    print("=" * 65)
    print("Device:", device)

    base_model = load_model(
        BASE_CHECKPOINT,
        device
    )

    pilot_model = load_model(
        PILOT_CHECKPOINT,
        device
    )

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

    near_flags = load_near_monochrome_flags()

    names = test_dataset.image_names

    expected_names = set(names)
    missing_flags = expected_names - set(near_flags)

    if missing_flags:
        raise RuntimeError(
            f"Missing audit results for {len(missing_flags)} test images."
        )

    groups = {
        "all": [],
        "color_rich": [],
        "near_monochrome_candidates": []
    }

    missing_ddcolor = []
    processed = 0

    with torch.inference_mode():
        for L_batch, target_AB_batch in test_loader:
            L_device = L_batch.to(
                device,
                non_blocking=True
            )

            base_batch = base_model(
                L_device
            ).cpu().numpy()

            pilot_batch = pilot_model(
                L_device
            ).cpu().numpy()

            L_numpy = L_batch.numpy()
            target_numpy = target_AB_batch.numpy()

            for i in range(L_batch.size(0)):
                name = names[processed]

                ddcolor_path = os.path.join(
                    DDCOLOR_DIR,
                    name
                )

                ddcolor_bgr = cv2.imread(
                    ddcolor_path
                )

                if ddcolor_bgr is None:
                    missing_ddcolor.append(name)
                    processed += 1
                    continue

                L = L_numpy[i, 0]
                target_AB = target_numpy[i]

                target_rgb = lab_to_rgb(
                    L,
                    target_AB
                )

                base_rgb = lab_to_rgb(
                    L,
                    base_batch[i]
                )

                pilot_rgb = lab_to_rgb(
                    L,
                    pilot_batch[i]
                )

                ddcolor_rgb = cv2.cvtColor(
                    ddcolor_bgr,
                    cv2.COLOR_BGR2RGB
                )

                ddcolor_rgb = cv2.resize(
                    ddcolor_rgb,
                    (IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=cv2.INTER_AREA
                )

                entry = {
                    "name": name,
                    "near_monochrome": near_flags[name],
                    "Original advanced": calculate_metrics(
                        base_rgb, target_rgb
                    ),
                    "Teacher-guided pilot": calculate_metrics(
                        pilot_rgb, target_rgb
                    ),
                    "DDColor": calculate_metrics(
                        ddcolor_rgb, target_rgb
                    )
                }

                groups["all"].append(entry)

                group_name = (
                    "near_monochrome_candidates"
                    if near_flags[name]
                    else "color_rich"
                )

                groups[group_name].append(entry)

                processed += 1

            if processed % 200 == 0:
                print(f"Processed {processed}/1000 images")

    report_lines = []

    print("\n" + "=" * 65)
    print("RESULTS BY IMAGE GROUP")
    print("=" * 65)

    for group_name, entries in groups.items():
        print(f"\n{group_name}: {len(entries)} evaluated images")
        report_lines.append(
            f"{group_name}: {len(entries)} evaluated images"
        )

        if not entries:
            print("No images in this group.")
            report_lines.append("No images in this group.")
            continue

        for model_name in [
            "Original advanced",
            "Teacher-guided pilot",
            "DDColor"
        ]:
            print(f"\n  {model_name}")
            report_lines.append(f"  {model_name}")

            for metric in ["MSE", "MAE", "PSNR", "SSIM"]:
                values = [
                    entry[model_name][metric]
                    for entry in entries
                ]

                mean_value = float(np.mean(values))

                line = f"    {metric}: {mean_value:.6f}"
                print(line)
                report_lines.append(line)

    if missing_ddcolor:
        print("\nMissing DDColor outputs:", len(missing_ddcolor))
        report_lines.append(
            f"Missing DDColor outputs: {len(missing_ddcolor)}"
        )

    report_path = os.path.join(
        OUTPUT_DIR,
        "grouped_metrics.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(report_lines))

    print("\nReport saved to:", report_path)
    print("Grouped evaluation completed.")


if __name__ == "__main__":
    main()