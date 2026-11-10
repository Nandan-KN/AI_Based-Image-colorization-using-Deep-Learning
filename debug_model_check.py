import cv2
import numpy as np
import torch
from pathlib import Path

from advanced_ddcolor_model import AdvancedDDColorInspired

image_dir = Path(r"C:\Users\Nandan\OneDrive\Documents\AI_Image_Colorization_Deploy\advanced_dataset\images")
img_path = next(image_dir.glob("*.jpg"), None)

if img_path is None:
    raise FileNotFoundError(f"No JPG file found in: {image_dir}")

img = cv2.imread(str(img_path))
if img is None:
    raise FileNotFoundError(f"Image could not be loaded: {img_path}")

img = cv2.resize(img, (256, 256))
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
L = lab[:, :, 0] / 255.0
L_t = torch.from_numpy(L).unsqueeze(0).unsqueeze(0)

model = AdvancedDDColorInspired(num_queries=16)
ck = torch.load(
    r"C:\Users\Nandan\OneDrive\Documents\AI_Image_Colorization_Deploy\models\advanced_ddcolor_best.pth",
    map_location="cpu"
)
model.load_state_dict(ck["model_state_dict"], strict=True)
model.eval()

with torch.inference_mode():
    out = model(L_t)

print("image path:", img_path)
print("shape:", tuple(out.shape))
print("min:", float(out.min()))
print("max:", float(out.max()))
print("mean:", float(out.mean()))