from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AdvancedSplitDataset(Dataset):

    def __init__(
        self,
        split_file,
        image_root="advanced_dataset/images",
        image_size=256
    ):
        self.image_root = Path(image_root)
        self.image_size = image_size

        split_path = Path(split_file)

        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}"
            )

        with open(
            split_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.image_names = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if len(self.image_names) == 0:
            raise RuntimeError(
                f"No images listed in: {split_path}"
            )

        # Verify every image exists
        missing = []

        for name in self.image_names:

            image_path = (
                self.image_root / name
            )

            if not image_path.exists():
                missing.append(name)

        if missing:

            raise RuntimeError(
                f"{len(missing)} images are missing. "
                f"First missing: {missing[:5]}"
            )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        image_name = self.image_names[index]

        image_path = (
            self.image_root / image_name
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (
                self.image_size,
                self.image_size
            ),
            interpolation=cv2.INTER_AREA
        )

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2LAB
        )

        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]

        L = (
            L.astype(np.float32)
            / 255.0
        )

        A = (
            A.astype(np.float32)
            - 128.0
        ) / 128.0

        B = (
            B.astype(np.float32)
            - 128.0
        ) / 128.0

        L = torch.from_numpy(
            L
        ).unsqueeze(0)

        AB = np.stack(
            [A, B],
            axis=0
        )

        AB = torch.from_numpy(
            AB.astype(np.float32)
        )

        return L, AB


def create_advanced_loaders(
    image_size=256,
    batch_size=2
):

    train_dataset = AdvancedSplitDataset(
        "advanced_dataset/splits/train.txt",
        image_size=image_size
    )

    validation_dataset = AdvancedSplitDataset(
        "advanced_dataset/splits/validation.txt",
        image_size=image_size
    )

    test_dataset = AdvancedSplitDataset(
        "advanced_dataset/splits/test.txt",
        image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader
    )


if __name__ == "__main__":

    print("=" * 60)
    print("ADVANCED SPLIT LOADER TEST")
    print("=" * 60)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader
    ) = create_advanced_loaders(
        image_size=256,
        batch_size=2
    )

    print(
        "Training dataset:",
        len(train_dataset)
    )

    print(
        "Validation dataset:",
        len(validation_dataset)
    )

    print(
        "Test dataset:",
        len(test_dataset)
    )

    L, AB = next(
        iter(train_loader)
    )

    print(
        "Training L shape:",
        L.shape
    )

    print(
        "Training AB shape:",
        AB.shape
    )

    print(
        "Advanced split loader test: PASSED"
    )

    print("=" * 60)