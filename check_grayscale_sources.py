import csv
from pathlib import Path

import cv2
import numpy as np


MANIFEST = Path("advanced_dataset/splits/train.txt")
IMAGE_DIR = Path("advanced_dataset/images")
OUTPUT_CSV = Path("advanced_outputs/grayscale_audit.csv")


def main():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        image_names = [
            line.strip()
            for line in f
            if line.strip()
        ]

    results = []
    failed = []

    print("Training images:", len(image_names))
    print("Checking grayscale content...")

    for index, name in enumerate(image_names, start=1):
        path = IMAGE_DIR / name

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            failed.append(name)
            continue

        # Downsample to make this audit faster.
        image = cv2.resize(
            image,
            (64, 64),
            interpolation=cv2.INTER_AREA
        )

        # Per-pixel difference between the largest
        # and smallest RGB/BGR channel.
        spread = (
            image.max(axis=2).astype(np.int16)
            - image.min(axis=2).astype(np.int16)
        )

        mean_spread = float(spread.mean())

        # Percentage of pixels with noticeable channel
        # differences. This indicates where color is present.
        colored_pixel_pct = float(
            np.mean(spread > 12) * 100
        )

        results.append({
            "filename": name,
            "mean_channel_spread": mean_spread,
            "colored_pixel_percentage": colored_pixel_pct,
            "near_monochrome": (
                mean_spread <= 5
                and colored_pixel_pct <= 5
            )
        })

        if index % 2000 == 0:
            print(f"Checked {index}/{len(image_names)}")

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "mean_channel_spread",
                "colored_pixel_percentage",
                "near_monochrome"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    near_mono = [
        row for row in results
        if row["near_monochrome"]
    ]

    print("\n" + "=" * 55)
    print("GRAYSCALE DATASET AUDIT")
    print("=" * 55)
    print("Manifest images:", len(image_names))
    print("Successfully checked:", len(results))
    print("Unreadable images:", len(failed))
    print("Near-monochrome candidates:", len(near_mono))
    print(
        "Near-monochrome percentage:",
        f"{100 * len(near_mono) / max(len(results), 1):.2f}%"
    )
    print("Full report:", OUTPUT_CSV)

    print("\nTen lowest-colorfulness candidates:")
    for row in sorted(
        results,
        key=lambda x: x["mean_channel_spread"]
    )[:10]:
        print(
            row["filename"],
            "| mean spread:",
            round(row["mean_channel_spread"], 2),
            "| colored pixels:",
            round(row["colored_pixel_percentage"], 2),
            "%"
        )


if __name__ == "__main__":
    main()