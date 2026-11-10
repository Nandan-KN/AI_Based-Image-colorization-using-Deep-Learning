from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AdvancedTeacherDataset(Dataset):

    def __init__(
        self,
        manifest="advanced_dataset/splits/train.txt",
        image_root="advanced_dataset/images",
        teacher_root="advanced_dataset/teacher_outputs/train",
        image_size=256
    ):
        self.image_root = Path(image_root)
        self.teacher_root = Path(teacher_root)
        self.image_size = image_size

        manifest_path = Path(manifest)

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}"
            )

        with open(
            manifest_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.image_names = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if not self.image_names:
            raise RuntimeError("Training manifest is empty.")

        # Verify original and teacher files are paired.
        missing_original = []
        missing_teacher = []

        for name in self.image_names:
            if not (self.image_root / name).is_file():
                missing_original.append(name)

            if not (self.teacher_root / name).is_file():
                missing_teacher.append(name)

        if missing_original or missing_teacher:
            raise RuntimeError(
                f"Missing originals: {len(missing_original)}; "
                f"missing teacher outputs: {len(missing_teacher)}. "
                f"First missing teacher files: {missing_teacher[:5]}"
            )

    def __len__(self):
        return len(self.image_names)

    def _rgb_to_lab_tensors(self, rgb_image):

        lab = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2LAB
        )

        L = (
            lab[:, :, 0].astype(np.float32) / 255.0
        )

        A = (
            lab[:, :, 1].astype(np.float32) - 128.0
        ) / 128.0

        B = (
            lab[:, :, 2].astype(np.float32) - 128.0
        ) / 128.0

        L_tensor = torch.from_numpy(
            L.copy()
        ).unsqueeze(0)

        AB = np.stack(
            [A, B],
            axis=0
        ).astype(np.float32)

        AB_tensor = torch.from_numpy(AB)

        return L_tensor, AB_tensor

    def __getitem__(self, index):

        name = self.image_names[index]

        original_path = self.image_root / name
        teacher_path = self.teacher_root / name

        original_bgr = cv2.imread(
            str(original_path),
            cv2.IMREAD_COLOR
        )

        teacher_bgr = cv2.imread(
            str(teacher_path),
            cv2.IMREAD_COLOR
        )

        if original_bgr is None:
            raise RuntimeError(
                f"Cannot read original image: {original_path}"
            )

        if teacher_bgr is None:
            raise RuntimeError(
                f"Cannot read teacher image: {teacher_path}"
            )

        original_rgb = cv2.cvtColor(
            original_bgr,
            cv2.COLOR_BGR2RGB
        )

        teacher_rgb = cv2.cvtColor(
            teacher_bgr,
            cv2.COLOR_BGR2RGB
        )

        size = (
            self.image_size,
            self.image_size
        )

        original_rgb = cv2.resize(
            original_rgb,
            size,
            interpolation=cv2.INTER_AREA
        )

        teacher_rgb = cv2.resize(
            teacher_rgb,
            size,
            interpolation=cv2.INTER_AREA
        )

        L, ground_truth_AB = self._rgb_to_lab_tensors(
            original_rgb
        )

        _, teacher_AB = self._rgb_to_lab_tensors(
            teacher_rgb
        )

        return L, ground_truth_AB, teacher_AB


def create_teacher_loader(
    batch_size=2,
    image_size=256
):

    dataset = AdvancedTeacherDataset(
        image_size=image_size
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return dataset, loader


if __name__ == "__main__":

    print("=" * 60)
    print("DDCOLOR TEACHER DATA LOADER TEST")
    print("=" * 60)

    dataset, loader = create_teacher_loader(
        batch_size=2,
        image_size=256
    )

    print("Training pairs:", len(dataset))

    L, ground_truth_AB, teacher_AB = next(
        iter(loader)
    )

    print("Input L shape:", L.shape)
    print("Ground-truth AB shape:", ground_truth_AB.shape)
    print("Teacher AB shape:", teacher_AB.shape)

    print(
        "Input L range:",
        L.min().item(),
        "to",
        L.max().item()
    )

    print(
        "Ground-truth AB range:",
        ground_truth_AB.min().item(),
        "to",
        ground_truth_AB.max().item()
    )

    print(
        "Teacher AB range:",
        teacher_AB.min().item(),
        "to",
        teacher_AB.max().item()
    )

    print(
        "Mean absolute teacher/ground-truth AB difference:",
        (teacher_AB - ground_truth_AB).abs().mean().item()
    )

    assert L.shape == (2, 1, 256, 256)
    assert ground_truth_AB.shape == (2, 2, 256, 256)
    assert teacher_AB.shape == (2, 2, 256, 256)

    assert torch.isfinite(L).all()
    assert torch.isfinite(ground_truth_AB).all()
    assert torch.isfinite(teacher_AB).all()

    print()
    print("Teacher data loader test: PASSED")
    print("=" * 60)