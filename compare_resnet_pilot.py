import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

from advanced_ddcolor_model import AdvancedDDColorInspired
from advanced_resnet_colorizer import ResNet18Colorizer
from advanced_split_loader import AdvancedSplitDataset


BASE_CHECKPOINT = "models/advanced_ddcolor_best.pth"
RESNET_CHECKPOINT = "models/resnet18_colorizer_pilot.pth"

TEST_MANIFEST = "advanced_dataset/splits/test.txt"
IMAGE_ROOT = "advanced_dataset/images"
DDCOLOR_DIR = "advanced_dataset/teacher_outputs/test"

OUTPUT_DIR = "advanced_outputs/resnet_pilot_comparison"

IMAGE_SIZE = 256
BATCH_SIZE = 2
NUM_VISUAL_SAMPLES = 10


def lab_to_rgb(L, AB):
    """Reconstruct RGB from normalized L and A/B channels."""

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


def calculate_metrics(prediction_rgb, target_rgb):
    prediction = (
        prediction_rgb.astype(np.float32) / 255.0
    )

    target = (
        target_rgb.astype(np.float32) / 255.0
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

    return {
        "MSE": mse,
        "MAE": mae,
        "PSNR": float(psnr),
        "SSIM": float(ssim)
    }


def load_advanced_model(device):
    checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location=device
    )

    model = AdvancedDDColorInspired(
        num_queries=checkpoint.get(
            "num_queries", 16
        )
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def load_resnet_model(device):
    checkpoint = torch.load(
        RESNET_CHECKPOINT,
        map_location=device
    )

    # Checkpoint already contains its complete encoder weights.
    model = ResNet18Colorizer(
        pretrained=False
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def save_comparison(name, panels):
    header_height = 32
    height, width = panels[0][1].shape[:2]

    canvas = np.zeros(
        (
            height + header_height,
            width * len(panels),
            3
        ),
        dtype=np.uint8
    )

    for index, (label, image_rgb) in enumerate(panels):
        x = index * width

        image_rgb = cv2.resize(
            image_rgb,
            (width, height),
            interpolation=cv2.INTER_AREA
        )

        canvas[
            header_height:, x:x + width
        ] = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR
        )

        cv2.putText(
            canvas,
            label,
            (x + 5, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        name
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
    print("RESNET-18 PILOT COMPARISON")
    print("=" * 65)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading original advanced model...")
    advanced_model = load_advanced_model(device)

    print("Loading ResNet-18 pilot...")
    resnet_model = load_resnet_model(device)

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

    print("Unseen test images:", len(test_dataset))

    metric_names = ["MSE", "MAE", "PSNR", "SSIM"]

    metrics = {
        "Original advanced model": {
            name: [] for name in metric_names
        },
        "ResNet-18 pilot": {
            name: [] for name in metric_names
        },
        "DDColor": {
            name: [] for name in metric_names
        }
    }

    processed = 0

    with torch.inference_mode():
        for L_batch, target_AB_batch in test_loader:

            L_device = L_batch.to(
                device,
                non_blocking=True
            )

            advanced_AB = advanced_model(
                L_device
            ).cpu().numpy()

            resnet_AB = resnet_model(
                L_device
            ).cpu().numpy()

            L_numpy = L_batch.numpy()
            target_numpy = target_AB_batch.numpy()

            for i in range(L_batch.size(0)):

                name = test_dataset.image_names[processed]

                ddcolor_path = os.path.join(
                    DDCOLOR_DIR,
                    name
                )

                ddcolor_bgr = cv2.imread(
                    ddcolor_path
                )

                if ddcolor_bgr is None:
                    raise FileNotFoundError(
                        f"DDColor output missing: {ddcolor_path}"
                    )

                L = L_numpy[i, 0]
                target_AB = target_numpy[i]

                target_rgb = lab_to_rgb(
                    L, target_AB
                )

                advanced_rgb = lab_to_rgb(
                    L, advanced_AB[i]
                )

                resnet_rgb = lab_to_rgb(
                    L, resnet_AB[i]
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

                predictions = {
                    "Original advanced model": advanced_rgb,
                    "ResNet-18 pilot": resnet_rgb,
                    "DDColor": ddcolor_rgb
                }

                for model_name, prediction in predictions.items():
                    result = calculate_metrics(
                        prediction,
                        target_rgb
                    )

                    for metric in metric_names:
                        metrics[model_name][metric].append(
                            result[metric]
                        )

                if processed < NUM_VISUAL_SAMPLES:

                    gray = np.clip(
                        L * 255.0, 0, 255
                    ).astype(np.uint8)

                    gray_rgb = np.repeat(
                        gray[..., None],
                        3,
                        axis=2
                    )

                    filename = os.path.splitext(name)[0]

                    save_comparison(
                        f"comparison_{processed + 1:03d}_{filename}.jpg",
                        [
                            ("Grayscale", gray_rgb),
                            ("Original advanced", advanced_rgb),
                            ("ResNet-18 pilot", resnet_rgb),
                            ("DDColor", ddcolor_rgb),
                            ("Ground truth", target_rgb)
                        ]
                    )

                processed += 1

            if processed % 200 == 0:
                print(
                    f"Evaluated {processed}/{len(test_dataset)}"
                )

    report_path = os.path.join(
        OUTPUT_DIR,
        "metrics.txt"
    )

    report_lines = []

    print("\n" + "=" * 65)
    print("FINAL TEST METRICS")
    print("=" * 65)

    print("Test images evaluated:", processed)

    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}")
        report_lines.append(model_name)

        for metric in metric_names:
            value = float(
                np.mean(model_metrics[metric])
            )

            print(f"{metric}: {value:.6f}")
            report_lines.append(
                f"{metric}: {value:.6f}"
            )

        report_lines.append("")

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(report_lines))

    print("\nVisual comparisons:", OUTPUT_DIR)
    print("Metrics report:", report_path)
    print("ResNet-18 comparison completed.")


if __name__ == "__main__":
    main()