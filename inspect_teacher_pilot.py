import os
import cv2
import numpy as np

MANIFEST = "advanced_dataset/splits/train.txt"
IMAGE_DIR = "advanced_dataset/images"
TEACHER_DIR = "advanced_dataset/teacher_outputs/train"
OUTPUT_PATH = "advanced_outputs/teacher_pilot_preview.jpg"

with open(MANIFEST, "r", encoding="utf-8") as f:
    image_name = next(
        line.strip() for line in f if line.strip()
    )

original_path = os.path.join(IMAGE_DIR, image_name)
teacher_path = os.path.join(TEACHER_DIR, image_name)

original = cv2.imread(original_path)
teacher = cv2.imread(teacher_path)

if original is None:
    raise FileNotFoundError(original_path)

if teacher is None:
    raise FileNotFoundError(teacher_path)

size = (256, 256)

# Resize both images to the same dimensions.
original = cv2.resize(original, size)
teacher = cv2.resize(teacher, size)

# Create a grayscale visualization from the original.
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

panels = [
    ("Grayscale input", gray),
    ("DDColor teacher", teacher),
    ("Original color", original),
]

header_height = 32
canvas = np.zeros((256 + header_height, 256 * 3, 3), dtype=np.uint8)

for i, (label, image) in enumerate(panels):
    x = i * 256
    canvas[header_height:, x:x + 256] = image

    cv2.putText(
        canvas,
        label,
        (x + 6, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
cv2.imwrite(OUTPUT_PATH, canvas)

print("Image:", image_name)
print("Preview saved to:", OUTPUT_PATH)