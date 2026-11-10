import torch
from torch.utils.data import DataLoader

from advanced_resnet_colorizer import ResNet18Colorizer
from advanced_split_loader import AdvancedSplitDataset
from advanced_color_loss import AdvancedColorLoss


def main():

    print("=" * 65)
    print("RESNET-18 COLORIZATION TRAINING TEST")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Load one real image from the fixed training split.
    dataset = AdvancedSplitDataset(
        split_file="advanced_dataset/splits/train.txt",
        image_root="advanced_dataset/images",
        image_size=256
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    L, target_AB = next(iter(loader))

    L = L.to(device)
    target_AB = target_AB.to(device)

    print("Training dataset:", len(dataset))
    print("Input shape:", L.shape)
    print("Target shape:", target_AB.shape)

    # Load the pretrained ResNet-18 model.
    model = ResNet18Colorizer(
        pretrained=True
    ).to(device)

    criterion = AdvancedColorLoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=1e-5
    )

    # Full-precision diagnostic: no AMP.
    model.train()
    optimizer.zero_grad(set_to_none=True)

    prediction = model(L)

    print("Prediction shape:", prediction.shape)
    print(
        "Prediction range:",
        prediction.min().item(),
        "to",
        prediction.max().item()
    )

    losses = criterion(
        prediction,
        target_AB
    )

    print("Color loss:", losses["color"].item())
    print("Chroma loss:", losses["chroma"].item())
    print("Edge loss:", losses["edge"].item())
    print("Total loss:", losses["total"].item())

    # Backward pass.
    losses["total"].backward()

    gradient_count = 0
    invalid_gradients = 0

    for parameter in model.parameters():

        if parameter.grad is not None:
            gradient_count += 1

            if not torch.isfinite(parameter.grad).all():
                invalid_gradients += 1

    print("Parameters with gradients:", gradient_count)
    print("Invalid gradient tensors:", invalid_gradients)

    assert prediction.shape == (1, 2, 256, 256)
    assert torch.isfinite(losses["total"]).item()
    assert invalid_gradients == 0

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    if device.type == "cuda":

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        print(f"Peak GPU memory: {peak_memory:.2f} MB")

    print()
    print("Forward pass: PASSED")
    print("Loss calculation: PASSED")
    print("Backward pass: PASSED")
    print("Optimizer update: PASSED")
    print("RESNET-18 TRAINING TEST: PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()