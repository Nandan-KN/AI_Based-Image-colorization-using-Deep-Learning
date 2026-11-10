import os
import cv2
import numpy as np
import torch

from advanced_ddcolor_model import AdvancedDDColorInspired
from advanced_split_loader import AdvancedSplitDataset


BASE_CHECKPOINT = "models/advanced_ddcolor_best.pth"
PILOT_CHECKPOINT = "models/advanced_teacher_pilot.pth"

TEST_MANIFEST = "advanced_dataset/splits/test.txt"
IMAGE_ROOT = "advanced_dataset/images"
DDCOLOR_DIR = "advanced_dataset/teacher_outputs/test"

OUTPUT_DIR = "advanced_outputs/visual_audit"

IMAGE_SIZE = 256
NUM_SAMPLES = 10


def lab_to_rgb(L, AB):
    """Convert normalized Lab channels into RGB."""

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


def save_comparison(filename, panels):
    header_height = 32
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    canvas = np.zeros(
        (height + header_height, width * len(panels), 3),
        dtype=np.uint8
    )

    for index, (label, image_rgb) in enumerate(panels):
        x = index * width

        image_rgb = cv2.resize(
            image_rgb,
            (width, height),
            interpolation=cv2.INTER_AREA
        )

        image_bgr = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR
        )

        canvas[
            header_height:, x:x + width
        ] = image_bgr

        cv2.putText(
            canvas,
            label,
            (x + 5, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
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
    print("VISUAL AUDIT: ADVANCED MODEL vs DDCOLOR")
    print("=" * 65)
    print("Device:", device)

    dataset = AdvancedSplitDataset(
        split_file=TEST_MANIFEST,
        image_root=IMAGE_ROOT,
        image_size=IMAGE_SIZE
    )

    base_model = load_model(
        BASE_CHECKPOINT,
        device
    )

    pilot_model = load_model(
        PILOT_CHECKPOINT,
        device
    )

    # Choose 10 evenly spaced entries from the fixed test manifest.
    indices = np.linspace(
        0,
        len(dataset) - 1,
        min(NUM_SAMPLES, len(dataset)),
        dtype=int
    )

    successful = 0
    missing_ddcolor = []

    with torch.inference_mode():

        for audit_number, index in enumerate(indices, start=1):

            L_tensor, target_AB_tensor = dataset[index]
            image_name = dataset.image_names[index]

            L_batch = L_tensor.unsqueeze(0).to(device)

            base_AB = base_model(
                L_batch
            )[0].cpu().numpy()

            pilot_AB = pilot_model(
                L_batch
            )[0].cpu().numpy()

            L = L_tensor[0].numpy()
            target_AB = target_AB_tensor.numpy()

            gray = np.clip(
                L * 255.0, 0, 255
            ).astype(np.uint8)

            gray_rgb = np.repeat(
                gray[..., None],
                3,
                axis=2
            )

            base_rgb = lab_to_rgb(
                L, base_AB
            )

            pilot_rgb = lab_to_rgb(
                L, pilot_AB
            )

            target_rgb = lab_to_rgb(
                L, target_AB
            )

            # Use the DDColor output generated for this
            # exact held-out test image.
            ddcolor_path = os.path.join(
                DDCOLOR_DIR,
                image_name
            )

            ddcolor_bgr = cv2.imread(
                ddcolor_path
            )

            if ddcolor_bgr is None:
                missing_ddcolor.append(image_name)
                continue

            ddcolor_rgb = cv2.cvtColor(
                ddcolor_bgr,
                cv2.COLOR_BGR2RGB
            )

            ddcolor_rgb = cv2.resize(
                ddcolor_rgb,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_AREA
            )

            panels = [
                ("Grayscale input", gray_rgb),
                ("Original advanced", base_rgb),
                ("Teacher-guided pilot", pilot_rgb),
                ("DDColor", ddcolor_rgb),
                ("Ground truth", target_rgb),
            ]

            safe_name = os.path.splitext(
                os.path.basename(image_name)
            )[0]

            output_filename = (
                f"audit_{audit_number:02d}_{safe_name}.jpg"
            )

            save_comparison(
                output_filename,
                panels
            )

            successful += 1

            print(
                f"Saved {successful}/{len(indices)}: "
                f"{output_filename}"
            )

    print()
    print("Successful comparisons:", successful)
    print("Missing DDColor outputs:", len(missing_ddcolor))

    if missing_ddcolor:
        print("First missing files:", missing_ddcolor[:5])

    print("Output folder:", OUTPUT_DIR)
    print("Visual audit completed.")


if __name__ == "__main__":
    main()