from pathlib import Path
import random


# ============================================================
# Configuration
# ============================================================

SOURCE_DIR = Path("advanced_dataset/images")

OUTPUT_DIR = Path("advanced_dataset/splits")

TRAIN_COUNT = 18000
VAL_COUNT = 1000
TEST_COUNT = 1000

SEED = 42

EXISTING_COCO_DIR = Path("val2017")


# ============================================================
# Collect images
# ============================================================

image_paths = sorted(
    [
        p
        for p in SOURCE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png"
        }
    ]
)

print("=" * 60)
print("ADVANCED DATASET SPLIT PREPARATION")
print("=" * 60)

print(
    "Images found:",
    len(image_paths)
)


# ============================================================
# Verify total count
# ============================================================

expected_total = (
    TRAIN_COUNT
    + VAL_COUNT
    + TEST_COUNT
)

if len(image_paths) != expected_total:

    raise RuntimeError(
        f"Expected exactly "
        f"{expected_total} images, "
        f"but found {len(image_paths)}."
    )


# ============================================================
# Check overlap with existing COCO val2017
# ============================================================

if EXISTING_COCO_DIR.exists():

    existing_names = {
        p.name
        for p in EXISTING_COCO_DIR.iterdir()
        if p.is_file()
    }

    overlap = [
        p.name
        for p in image_paths
        if p.name in existing_names
    ]

    print(
        "Overlap with existing val2017:",
        len(overlap)
    )

    if overlap:

        print(
            "First overlapping files:"
        )

        for name in overlap[:10]:
            print(
                " ",
                name
            )

        raise RuntimeError(
            "Dataset overlap detected. "
            "No split was created."
        )

else:

    print(
        "Existing val2017 folder not found."
    )


# ============================================================
# Shuffle reproducibly
# ============================================================

random.seed(SEED)

random.shuffle(
    image_paths
)


# ============================================================
# Create splits
# ============================================================

train_paths = image_paths[
    :TRAIN_COUNT
]

val_paths = image_paths[
    TRAIN_COUNT:
    TRAIN_COUNT + VAL_COUNT
]

test_paths = image_paths[
    TRAIN_COUNT + VAL_COUNT:
]


# ============================================================
# Create output directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Save manifests
# ============================================================

def save_manifest(
    paths,
    filename
):

    output_file = (
        OUTPUT_DIR
        / filename
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for path in paths:

            f.write(
                path.name
                + "\n"
            )

    return output_file


train_file = save_manifest(
    train_paths,
    "train.txt"
)

val_file = save_manifest(
    val_paths,
    "validation.txt"
)

test_file = save_manifest(
    test_paths,
    "test.txt"
)


# ============================================================
# Verify no overlap between splits
# ============================================================

train_names = {
    p.name
    for p in train_paths
}

val_names = {
    p.name
    for p in val_paths
}

test_names = {
    p.name
    for p in test_paths
}


assert train_names.isdisjoint(
    val_names
)

assert train_names.isdisjoint(
    test_names
)

assert val_names.isdisjoint(
    test_names
)


# ============================================================
# Final report
# ============================================================

print()
print(
    "Training images:",
    len(train_paths)
)

print(
    "Validation images:",
    len(val_paths)
)

print(
    "Test images:",
    len(test_paths)
)

print()
print(
    "Train manifest:",
    train_file
)

print(
    "Validation manifest:",
    val_file
)

print(
    "Test manifest:",
    test_file
)

print()
print(
    "Advanced dataset split: PASSED"
)

print("=" * 60)