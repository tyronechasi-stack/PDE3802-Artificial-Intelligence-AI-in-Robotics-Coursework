"""Sketch 2: teach the last layer of MobileNet our six item names."""

import argparse
import csv
import random
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from prepare_data import CLASSES

ROOT = Path(__file__).resolve().parent
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001


def image_transform(training=False):
    # Make every image the same size. Only training gets small random changes.
    steps = [transforms.Resize((224, 224))]
    if training:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    steps += [transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(steps)


def make_model(pretrained=False):
    # Reuse learned features. Only the new final layer learns our six classes.
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    inputs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(inputs, len(CLASSES))
    return model


def run_epoch(model, loader, loss_function, optimizer=None):
    # An optimizer means training. Without one, we only check the model.
    training = optimizer is not None
    model.eval()  # Keep the frozen feature layers unchanged.
    model.classifier.train(training)
    actual, predicted = [], []
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            if training:
                optimizer.zero_grad()  # Clear the old gradients.
            scores = model(images)  # Raw class scores, called logits.
            loss = loss_function(scores, labels)
            if training:
                loss.backward()  # Calculate how the weights should change.
                optimizer.step()  # Update the new final layer.
            total_loss += loss.item() * len(labels)
            actual.extend(labels.tolist())
            predicted.extend(scores.argmax(dim=1).tolist())
    accuracy = sum(a == p for a, p in zip(actual, predicted)) / len(actual)
    macro_f1 = f1_score(actual, predicted, labels=range(6), average="macro", zero_division=0)
    return total_loss / len(actual), accuracy, macro_f1


def train_model(epochs=EPOCHS):
    # 1. Repeat the same random choices when possible. We use the CPU for simplicity.
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    train = datasets.ImageFolder(ROOT / "data/split/train", image_transform(True))
    val = datasets.ImageFolder(ROOT / "data/split/val", image_transform())
    if train.classes != CLASSES or val.classes != CLASSES:
        raise ValueError("The train and val folders must contain the six numbered classes.")
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, num_workers=0)

    # 2. Get pretrained weights once. The first run needs an internet connection.
    model = make_model(pretrained=True)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier[-1].parameters(), lr=LEARNING_RATE)
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    best_f1 = -1.0
    history = []

    # 3. Learn from train; select the best model using val. Do not open test here.
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _ = run_epoch(model, train_loader, loss_function, optimizer)
        _, val_acc, val_f1 = run_epoch(model, val_loader, loss_function)
        history.append([epoch, train_loss, train_acc, val_acc, val_f1])
        print(f"Epoch {epoch}/{epochs} | train accuracy {train_acc:.3f} | "
              f"val accuracy {val_acc:.3f} | val macro-F1 {val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"weights": model.state_dict(), "classes": train.classes,
                        "epoch": epoch, "validation_macro_f1": val_f1},
                       output / "best_model.pt")

    # 4. Save the learning record so we can discuss progress with the lecturer.
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_accuracy", "val_macro_f1"])
        writer.writerows(history)
    print("Saved outputs/best_model.pt. Next: python test_model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    try:
        train_model(args.epochs)
    except (ValueError, OSError) as error:
        raise SystemExit(f"Cannot train: {error}")
