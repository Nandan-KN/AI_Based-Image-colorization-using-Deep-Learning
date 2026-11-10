import torch
import torch.nn as nn

from advanced_ddcolor_model import AdvancedDDColorInspired
from advanced_dataset_loader import create_advanced_dataloader


def main():

    print()
    print("=" * 60)
    print("ADVANCED MODEL ONE-BATCH TRAINING TEST")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset, loader = create_advanced_dataloader(
        image_dir="advanced_dataset/images",
        batch_size=2,
        image_size=256
    )

    L, target_AB = next(iter(loader))

    L = L.to(
        device,
        non_blocking=True
    )

    target_AB = target_AB.to(
        device,
        non_blocking=True
    )

    print(
        "Input shape:",
        L.shape
    )

    print(
        "Target shape:",
        target_AB.shape
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AdvancedDDColorInspired(
        num_queries=16
    ).to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    predicted_AB = model(L)

    print(
        "Prediction shape:",
        predicted_AB.shape
    )

    print(
        "Prediction range:",
        predicted_AB.min().item(),
        "to",
        predicted_AB.max().item()
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss = criterion(
        predicted_AB,
        target_AB
    )

    print(
        "Initial loss:",
        loss.item()
    )

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # Check gradients
    # --------------------------------------------------------

    gradient_count = 0
    invalid_gradients = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_count += 1

            if not torch.isfinite(
                parameter.grad
            ).all():

                invalid_gradients += 1

    print(
        "Parameters with gradients:",
        gradient_count
    )

    print(
        "Invalid gradient tensors:",
        invalid_gradients
    )

    # --------------------------------------------------------
    # Optimizer update
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    if torch.cuda.is_available():

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        reserved_memory = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 2)
        )

        print(
            "Peak GPU memory:",
            f"{peak_memory:.2f} MB"
        )

        print(
            "GPU memory reserved:",
            f"{reserved_memory:.2f} MB"
        )

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    assert predicted_AB.shape == (
        2,
        2,
        256,
        256
    )

    assert torch.isfinite(
        loss
    ).item()

    assert invalid_gradients == 0

    print()
    print(
        "One-batch forward pass: PASSED"
    )

    print(
        "Loss calculation: PASSED"
    )

    print(
        "Backward pass: PASSED"
    )

    print(
        "Optimizer update: PASSED"
    )

    print()
    print(
        "ADVANCED MODEL ONE-BATCH TEST: PASSED"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()