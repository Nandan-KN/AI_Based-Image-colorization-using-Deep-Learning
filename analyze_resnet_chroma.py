import csv

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from advanced_ddcolor_model import AdvancedDDColorInspired
from advanced_resnet_colorizer import ResNet18Colorizer
from advanced_split_loader import AdvancedSplitDataset


TEST_MANIFEST = "advanced_dataset/splits/test.txt"
IMAGE_ROOT = "advanced_dataset/images"
DDCOLOR_DIR = "advanced_dataset/teacher_outputs/test"
AUDIT_CSV = "advanced_outputs/evaluation_grayscale_audit.csv"

BASE_CHECKPOINT = "models/advanced_ddcolor_best.pth"
RESNET_CHECKPOINT = "models/resnet18_colorizer_finetuned.pth"

BATCH_SIZE = 2
IMAGE_SIZE = 256


def mean_chroma(ab):
    """Mean chroma strength of normalized A/B channels."""
    chroma = np.sqrt(
        ab[0] ** 2 + ab[1] ** 2
    )
    return float(chroma.mean()), float(np.percentile(chroma, 90))


def load_flags():
    flags = {}

    with open(AUDIT_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"].strip().lower() == "test":
                flags[row["filename"]] = (
                    row["near_monochrome"].strip().lower() == "true"
                )

    return flags


def load_advanced(device):
    checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location=device
    )

    model = AdvancedDDColorInspired(
        num_queries=checkpoint.get("num_queries", 16)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_resnet(device):
    checkpoint = torch.load(
        RESNET_CHECKPOINT,
        map_location=device
    )

    model = ResNet18Colorizer(
        pretrained=False
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def ddcolor_to_ab(filename):
    path = f"{DDCOLOR_DIR}/{filename}"
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    a = (lab[:, :, 1].astype(np.float32) - 128.0) / 128.0
    b = (lab[:, :, 2].astype(np.float32) - 128.0) / 128.0

    return np.stack([a, b], axis=0)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("CHROMA STRENGTH ANALYSIS")
    print("=" * 65)
    print("Device:", device)

    dataset = AdvancedSplitDataset(
        split_file=TEST_MANIFEST,
        image_root=IMAGE_ROOT,
        image_size=IMAGE_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    flags = load_flags()
    models = {
        "Original advanced": load_advanced(device),
        "Fine-tuned ResNet-18": load_resnet(device)
    }

    stats = {
        group: {
            name: {"mean": [], "p90": []}
            for name in [
                "Ground truth",
                "Original advanced",
                "Fine-tuned ResNet-18",
                "DDColor"
            ]
        }
        for group in ["All test images", "Color-rich candidates"]
    }

    processed = 0

    with torch.inference_mode():
        for L_batch, target_AB_batch in loader:

            L_device = L_batch.to(device)

            predictions = {
                name: model(L_device).cpu().numpy()
                for name, model in models.items()
            }

            L_numpy = L_batch.numpy()
            target_numpy = target_AB_batch.numpy()

            for i in range(L_batch.size(0)):
                filename = dataset.image_names[processed]

                if filename not in flags:
                    raise RuntimeError(
                        f"Missing grayscale audit flag for {filename}"
                    )

                near_mono = flags[filename]

                ab_images = {
                    "Ground truth": target_numpy[i],
                    "Original advanced": predictions[
                        "Original advanced"
                    ][i],
                    "Fine-tuned ResNet-18": predictions[
                        "Fine-tuned ResNet-18"
                    ][i],
                    "DDColor": ddcolor_to_ab(filename)
                }

                groups = ["All test images"]

                if not near_mono:
                    groups.append("Color-rich candidates")

                for group in groups:
                    for name, ab in ab_images.items():
                        mean_value, p90_value = mean_chroma(ab)

                        stats[group][name]["mean"].append(mean_value)
                        stats[group][name]["p90"].append(p90_value)

                processed += 1

            if processed % 200 == 0:
                print(f"Processed {processed}/1000 images")

    for group, results in stats.items():
        print("\n" + "=" * 65)
        print(group)
        print("=" * 65)
        print(f"{'Model':25s} {'Mean chroma':>14s} {'90th percentile':>18s}")

        for name, values in results.items():
            mean_value = np.mean(values["mean"])
            p90_value = np.mean(values["p90"])

            print(
                f"{name:25s} {mean_value:14.5f} {p90_value:18.5f}"
            )

    print("\nChroma analysis completed.")


if __name__ == "__main__":
    main()