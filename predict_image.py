"""Sketch 4: load one photo and print its predicted item and confidence."""

import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image

from prepare_data import CLASSES
from train_model import image_transform, make_model

ROOT = Path(__file__).resolve().parent


def load_classifier():
    # 1. Load our trained weights. No pretrained download is needed here.
    checkpoint_path = ROOT / "outputs/best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Run train_model.py first to create outputs/best_model.pt.")
    torch.set_num_threads(4)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != CLASSES:
        raise ValueError("This checkpoint does not match the six classes in the report.")
    model = make_model()
    model.load_state_dict(checkpoint["weights"])
    model.eval()  # We are making predictions, not learning.
    return model, checkpoint["classes"], image_transform()


def classify_frame(frame_bgr, model, transform):
    # 2. OpenCV uses BGR. Our neural network uses RGB.
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = transform(image).unsqueeze(0)  # Add a batch of one image.
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)
    confidence, index = probabilities[0].max(dim=0)
    return index.item(), confidence.item()


def predict_image(image_path, output_path):
    # 3. Read a file, then reuse the same prediction steps as the webcam.
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Cannot read image: {image_path}")
    model, classes, transform = load_classifier()
    index, confidence = classify_frame(frame, model, transform)
    name = classes[index].split("_", 1)[1].replace("_", " ")
    message = f"{name}: {confidence:.1%}"
    print(message)

    # 4. Save a labelled photo. Confidence is a model score, not a guarantee.
    cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise OSError(f"Could not save {output_path}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a JPG or PNG with one main item")
    parser.add_argument("--output", default=str(ROOT / "outputs/prediction.jpg"))
    args = parser.parse_args()
    try:
        predict_image(args.image, args.output)
    except (ValueError, OSError) as error:
        raise SystemExit(f"Cannot predict: {error}")
