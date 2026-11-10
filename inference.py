from pathlib import Path

import cv2
import numpy as np
import torch

from cnn_model import CNNColorizationModel

def colorize_image(input_path, output_path, model_path="checkpoints/best_model.pt", image_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0

    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).to(device).float()

    model = CNNColorizationModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred_ab = model(L_tensor)

    pred_ab = pred_ab[0].cpu().numpy()
    pred_ab = pred_ab.transpose(1, 2, 0)
    pred_ab = (pred_ab * 128.0) + 128.0

    output_lab = np.zeros_like(lab, dtype=np.float32)
    output_lab[:, :, 0] = lab[:, :, 0]
    output_lab[:, :, 1:] = pred_ab

    output_rgb = cv2.cvtColor(output_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output_rgb)
    print(f"Saved colorized image to: {output_path}")
    return output_rgb

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Colorize a single image using the trained CNN model.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the colorized image.")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt", help="Path to the trained model checkpoint.")
    args = parser.parse_args()

    colorize_image(args.input, args.output, args.model)