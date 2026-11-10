import csv
from pathlib import Path

import cv2
import numpy as np


IMAGE_DIR = Path("advanced_dataset/images")

MANIFESTS = {
    "validation": Path("advanced_dataset/splits/validation.txt"),
    "test": Path("advanced_dataset/splits/test.txt"),
}

OUTPUT_CSV = Path(
    "advanced_outputs/evaluation_grayscale_audit.csv"
)


def audit_split(split_name, manifest_path):

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        image_names = [
            line.strip()
            for line in f
            if line.strip()
        ]

    results = []
    unreadable = []

    print(f"\nChecking {split_name}: {len(image_names)} images")

    for index, name in enumerate(image_names, start=1):

        image_path = IMAGE_DIR / name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            unreadable.append(name)
            continue

        image = cv2.resize(
            image,
            (64, 64),
            interpolation=cv2.INTER_AREA
        )

        spread = (
            image.max(axis=2).astype(np.int16)
            - image.min(axis=2).astype(np.int16)
        )

        mean_spread = float(spread.mean())

        colored_pixel_pct = float(
            np.mean(spread > 12) * 100
        )

        near_monochrome = (
            mean_spread <= 5
            and colored_pixel_pct <= 5
        )

        results.append({
            "split": split_name,
            "filename": name,
            "mean_channel_spread": mean_spread,
            "colored_pixel_percentage": colored_pixel_pct,
            "near_monochrome": near_monochrome,
        })

        if index % 250 == 0:
            print(f"Checked {index}/{len(image_names)}")

    near_mono_count = sum(
        row["near_monochrome"] for row in results
    )

    print(f"{split_name} images checked: {len(results)}")
    print(f"Near-monochrome candidates: {near_mono_count}")
    print(f"Unreadable images: {len(unreadable)}")

    return results


def main():

    all_results = []

    for split_name, manifest_path in MANIFESTS.items():
        all_results.extend(
            audit_split(split_name, manifest_path)
        )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fields = [
        "split",
        "filename",
        "mean_channel_spread",
        "colored_pixel_percentage",
        "near_monochrome",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 55)
    print("EVALUATION DATASET AUDIT COMPLETED")
    print("=" * 55)
    print("Total images checked:", len(all_results))
    print("Report saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
    