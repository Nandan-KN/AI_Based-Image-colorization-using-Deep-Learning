import torch

from advanced_ddcolor_model import AdvancedDDColorInspired
from advanced_split_loader import create_advanced_loaders
from advanced_color_loss import AdvancedColorLoss


def main():

    print()
    print("=" * 70)
    print("ADVANCED FULL ONE-BATCH INTEGRATION TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

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

    print()
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

    # --------------------------------------------------------
    # One real training batch
    # --------------------------------------------------------

    L, target_AB = next(
        iter(train_loader)
    )

    L = L.to(
        device,
        non_blocking=True
    )

    target_AB = target_AB.to(
        device,
        non_blocking=True
    )

    print()
    print(
        "Input L shape:",
        L.shape
    )

    print(
        "Target AB shape:",
        target_AB.shape
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AdvancedDDColorInspired(
        num_queries=16
    ).to(device)

    # --------------------------------------------------------
    # Advanced loss
    # --------------------------------------------------------

    criterion = AdvancedColorLoss(
        color_weight=1.0,
        chroma_weight=1.5,
        edge_weight=0.15
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    # --------------------------------------------------------
    # Training mode
    # --------------------------------------------------------

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    predicted_AB = model(
        L
    )

    print()
    print(
        "Predicted AB shape:",
        predicted_AB.shape
    )

    print(
        "Predicted AB range:",
        predicted_AB.min().item(),
        "to",
        predicted_AB.max().item()
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    losses = criterion(
        predicted_AB,
        target_AB
    )

    total_loss = losses["total"]

    print()
    print(
        "Color loss:",
        losses["color"].item()
    )

    print(
        "Chroma loss:",
        losses["chroma"].item()
    )

    print(
        "Edge loss:",
        losses["edge"].item()
    )

    print(
        "Total loss:",
        total_loss.item()
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    total_loss.backward()

    # --------------------------------------------------------
    # Gradient verification
    # --------------------------------------------------------

    parameters_with_gradients = 0
    invalid_gradients = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            parameters_with_gradients += 1

            if not torch.isfinite(
                parameter.grad
            ).all():

                invalid_gradients += 1

    print()
    print(
        "Parameters with gradients:",
        parameters_with_gradients
    )

    print(
        "Invalid gradient tensors:",
        invalid_gradients
    )

    # --------------------------------------------------------
    # Optimizer update
    # --------------------------------------------------------

    optimizer.step()

    print(
        "Optimizer update completed."
    )

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / (1024 ** 2)
        )

        reserved = (
            torch.cuda.memory_reserved()
            / (1024 ** 2)
        )

        peak = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        print()
        print(
            "GPU memory allocated:",
            f"{allocated:.2f} MB"
        )

        print(
            "GPU memory reserved:",
            f"{reserved:.2f} MB"
        )

        print(
            "Peak GPU memory:",
            f"{peak:.2f} MB"
        )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert predicted_AB.shape == (
        2,
        2,
        256,
        256
    )

    assert torch.isfinite(
        total_loss
    ).item()

    assert invalid_gradients == 0

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print()
    print(
        "Forward pass: PASSED"
    )

    print(
        "Advanced loss: PASSED"
    )

    print(
        "Backward pass: PASSED"
    )

    print(
        "Gradient check: PASSED"
    )

    print(
        "Optimizer update: PASSED"
    )

    print()
    print(
        "ADVANCED FULL ONE-BATCH TEST: PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()