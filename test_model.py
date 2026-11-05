"""Sketch 3: measure our selected model using only the held-out test photos."""

import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # Save the chart without opening a desktop window.
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)

from predict_image import classify_frame, load_classifier

ROOT = Path(__file__).resolve().parent


def test_model():
    # 1. Load the model already selected using validation data.
    model, classes, transform = load_classifier()
    names = [name.split("_", 1)[1].replace("_", " ") for name in classes]
    actual, predicted, rows = [], [], []
    for class_id, class_folder in enumerate(classes):
        folder = ROOT / "data/split/test" / class_folder
        photos = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not photos:
            raise ValueError(f"No test images in {folder}. Run prepare_data.py first.")
        for photo in photos:
            frame = cv2.imread(str(photo))
            if frame is None:
                raise ValueError(f"Cannot read test image: {photo}")
            guess, confidence = classify_frame(frame, model, transform)
            actual.append(class_id)
            predicted.append(guess)
            rows.append([str(photo.relative_to(ROOT)), names[class_id], names[guess], confidence])

    # 2. Accuracy counts correct answers. Macro-F1 gives each class equal weight.
    labels = list(range(len(classes)))
    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=labels)
    output = ROOT / "outputs/test_results"
    output.mkdir(parents=True, exist_ok=True)
    metrics = {"test_images": len(actual), "accuracy": float(accuracy), "macro_f1": float(macro_f1)}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(actual, predicted, labels=labels,
                                   target_names=names, zero_division=0)
    (output / "class_report.txt").write_text(report, encoding="utf-8")

    # 3. Save all guesses and the confusion matrix so we can inspect mistakes.
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["photo", "actual", "predicted", "confidence"])
        writer.writerows(rows)
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual / predicted"] + names)
        for name, values in zip(names, matrix.tolist()):
            writer.writerow([name] + values)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=names).plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 4. List mistakes. We still need to open the photos and explain the causes.
    mistakes = [row for row in rows if row[1] != row[2]]
    text = ["# Error analysis", "", f"Wrong predictions: {len(mistakes)} of {len(rows)}.",
            "", "Open these photos and note lighting, background, blur or similar shapes.",
            "Do not tune the model on the test set. Use validation data for changes.", ""]
    text += [f"- {row[0]}: actual {row[1]}, predicted {row[2]} ({row[3]:.1%})" for row in mistakes]
    (output / "error_analysis.md").write_text("\n".join(text), encoding="utf-8")
    print(f"Test accuracy: {accuracy:.3f}\nTest macro-F1: {macro_f1:.3f}")
    print(f"Saved test results to {output}")


if __name__ == "__main__":
    try:
        test_model()
    except (ValueError, OSError) as error:
        raise SystemExit(f"Cannot test: {error}")
