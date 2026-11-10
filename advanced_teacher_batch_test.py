import torch

from advanced_ddcolor_model import AdvancedDDColorInspired
from advanced_teacher_loader import create_teacher_loader
from advanced_teacher_loss import TeacherGuidedColorLoss


CHECKPOINT_PATH = "models/advanced_ddcolor_best.pth"
BATCH_SIZE = 2


def main():

    print("=" * 65)
    print("REAL TEACHER-GUIDED BATCH TEST")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = device.type == "cuda"

    print("Device:", device)

    if use_amp:
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Load the existing best checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model = AdvancedDDColorInspired(
        num_queries=checkpoint.get("num_queries", 16)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Loaded checkpoint epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["best_validation_loss"])

    # Load one real original/teacher batch.
    dataset, loader = create_teacher_loader(
        batch_size=BATCH_SIZE,
        image_size=256
    )

    L, ground_truth_AB, teacher_AB = next(iter(loader))

    L = L.to(device, non_blocking=True)
    ground_truth_AB = ground_truth_AB.to(
        device, non_blocking=True
    )
    teacher_AB = teacher_AB.to(device, non_blocking=True)

    print("Training pairs available:", len(dataset))
    print("L shape:", L.shape)
    print("Ground-truth AB shape:", ground_truth_AB.shape)
    print("Teacher AB shape:", teacher_AB.shape)

    # Teacher-guided objective.
    criterion = TeacherGuidedColorLoss(
        teacher_weight=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=1e-5
    )

  

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Forward pass and combined loss.
    # Run the model and loss in full precision.
    predicted_AB = model(L)

    losses = criterion(
    predicted_AB.float(),
    ground_truth_AB.float(),
    teacher_AB.float()
    )
    print("Prediction shape:", predicted_AB.shape)

    for name, value in losses.items():
        print(f"{name.capitalize()} loss: {value.item():.6f}")

    # Backpropagation.
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

    assert predicted_AB.shape == (2, 2, 256, 256)
    assert torch.isfinite(losses["total"]).item()
    assert invalid_gradients == 0

    # One in-memory optimizer update.
    optimizer.step()

    if use_amp:
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU memory: {peak_mb:.2f} MB")

    print()
    print("Real-data forward pass: PASSED")
    print("Teacher-guided loss: PASSED")
    print("Backward pass: PASSED")
    print("Optimizer update: PASSED")
    print("Existing checkpoint remains unchanged.")
    print("REAL TEACHER-GUIDED BATCH TEST: PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()