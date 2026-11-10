import math
from pathlib import Path

import cv2
import numpy as np
import torch

from advanced_ddcolor_model import AdvancedDDColorInspired
from advanced_split_loader import create_advanced_loaders


BASELINE_PATH = "models/advanced_ddcolor_best.pth"
TEACHER_PATH = "models/advanced_ddcolor_teacher_epoch1.pth"

OUTPUT_DIR = Path("advanced_outputs/checkpoint_comparison")
NUM_SAMPLES_TO_SAVE = 12
BATCH_SIZE = 2


def load_model(checkpoint_path, device):
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        str(path),
        map_location=device
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model = AdvancedDDColorInspired(
        num_queries=16
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model


def lab_to_rgb(L, AB):
    """Reconstruct an RGB image from normalized L and A/B channels."""

    lab = np.zeros(
        (L.shape[0], L.shape[1], 3),
        dtype=np.float32
    )

    lab[:, :, 0] = L * 255.0
    lab[:, :, 1] = AB[0] * 128.0 + 128.0
    lab[:, :, 2] = AB[1] * 128.0 + 128.0

    lab = np.clip(
        np.rint(lab),
        0,
        255
    ).astype(np.uint8)

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )


def calculate_metrics(prediction, target):
    pred = prediction.astype(np.float32) / 255.0
    truth = target.astype(np.float32) / 255.0

    difference = pred - truth

    mse = float(np.mean(difference ** 2))
    mae = float(np.mean(np.abs(difference)))

    psnr = (
        float("inf")
        if mse == 0
        else 10.0 * math.log10(1.0 / mse)
    )

    # Standard Gaussian-window SSIM approximation, averaged
    # across RGB channels.
    mu_pred = cv2.GaussianBlur(
        pred, (11, 11), 1.5
    )
    mu_truth = cv2.GaussianBlur(
        truth, (11, 11), 1.5
    )

    pred_sq = cv2.GaussianBlur(
        pred * pred, (11, 11), 1.5
    )
    truth_sq = cv2.GaussianBlur(
        truth * truth, (11, 11), 1.5
    )
    pred_truth = cv2.GaussianBlur(
        pred * truth, (11, 11), 1.5
    )

    var_pred = pred_sq - mu_pred * mu_pred
    var_truth = truth_sq - mu_truth * mu_truth
    covariance = pred_truth - mu_pred * mu_truth

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (
        (2 * mu_pred * mu_truth + c1)
        * (2 * covariance + c2)
    )

    denominator = (
        (mu_pred ** 2 + mu_truth ** 2 + c1)
        * (var_pred + var_truth + c2)
    )

    ssim = float(
        np.mean(
            numerator / (denominator + 1e-12)
        )
    )

    return {
        "MSE": mse,
        "MAE": mae,
        "PSNR": psnr,
        "SSIM": ssim
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("ADVANCED CHECKPOINT COMPARISON")
    print("=" * 65)
    print("Device:", device)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Both models use exactly the same held-out test images.
    (
        _train_dataset,
        _validation_dataset,
        test_dataset,
        _train_loader,
        _validation_loader,
        test_loader
    ) = create_advanced_loaders(
        image_size=256,
        batch_size=BATCH_SIZE
    )

    print("Test images:", len(test_dataset))

    baseline = load_model(
        BASELINE_PATH, device
    )

    teacher = load_model(
        TEACHER_PATH, device
    )

    results = {
        "Original advanced model": [],
        "Teacher-fine-tuned model": []
    }

    samples_saved = 0

    with torch.inference_mode():
        for batch_index, (L, target_AB) in enumerate(test_loader):

            L = L.to(device)
            target_AB = target_AB.to(device)

            baseline_AB = baseline(L)
            teacher_AB = teacher(L)

            for i in range(L.shape[0]):

                l_np = L[i, 0].cpu().numpy()
                target_np = target_AB[i].cpu().numpy()

                target_rgb = lab_to_rgb(
                    l_np,
                    target_np
                )

                baseline_rgb = lab_to_rgb(
                    l_np,
                    baseline_AB[i].cpu().numpy()
                )

                teacher_rgb = lab_to_rgb(
                    l_np,
                    teacher_AB[i].cpu().numpy()
                )

                results["Original advanced model"].append(
                    calculate_metrics(
                        baseline_rgb, target_rgb
                    )
                )

                results["Teacher-fine-tuned model"].append(
                    calculate_metrics(
                        teacher_rgb, target_rgb
                    )
                )

                if samples_saved < NUM_SAMPLES_TO_SAVE:

                    grayscale_rgb = np.repeat(
                        np.clip(
                            np.rint(l_np * 255),
                            0,
                            255
                        ).astype(np.uint8)[:, :, None],
                        3,
                        axis=2
                    )

                    panels = [
                        grayscale_rgb,
                        target_rgb,
                        baseline_rgb,
                        teacher_rgb
                    ]

                    panel_names = [
                        "Input",
                        "Ground truth",
                        "Original model",
                        "Teacher model"
                    ]

                    labeled = []

                    for panel, name in zip(
                        panels, panel_names
                    ):
                        panel_bgr = cv2.cvtColor(
                            panel,
                            cv2.COLOR_RGB2BGR
                        )

                        cv2.putText(
                            panel_bgr,
                            name,
                            (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

                        labeled.append(panel_bgr)

                    comparison = np.concatenate(
                        labeled,
                        axis=1
                    )

                    output_path = (
                        OUTPUT_DIR
                        / f"sample_{samples_saved + 1:03d}.jpg"
                    )

                    cv2.imwrite(
                        str(output_path),
                        comparison
                    )

                    samples_saved += 1

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Evaluated "
                    f"{min((batch_index + 1) * BATCH_SIZE, len(test_dataset))}"
                    f"/{len(test_dataset)} images"
                )

    print()
    print("FINAL TEST METRICS")
    print("-" * 65)

    for model_name, image_metrics in results.items():

        print()
        print(model_name)

        for metric_name in ("MSE", "MAE", "PSNR", "SSIM"):
            average = float(
                np.mean([
                    item[metric_name]
                    for item in image_metrics
                ])
            )

            print(
                f"{metric_name}: {average:.6f}"
            )

    print()
    print("Comparison images saved to:", OUTPUT_DIR)
    print("Checkpoint comparison completed.")


if __name__ == "__main__":
    main()