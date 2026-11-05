# Six class office item classifier

This is a small Part B project for one main item in a photo or webcam frame.
The classes come from `sources/Restarted_Y3C1_Updated.docx`.
Read `VIVA_GUIDE.docx` for a short explanation you can practise with your lecturer.

## The five Python sketches

| Run order | File | What it does |
|---|---|---|
| 1 | `prepare_data.py` | Checks the photos and splits whole item groups |
| 2 | `train_model.py` | Trains the new last layer of MobileNetV3 Small |
| 3 | `test_model.py` | Saves accuracy, macro-F1, a confusion matrix and mistakes |
| 4 | `predict_image.py` | Predicts one photo and saves a labelled copy |
| 5 | `live_camera.py` | Predicts successive webcam frames |

These are ordinary `.py` files. Keep them together in the extracted folder.
Run them from there in VS Code or PyCharm; do not move them into an IDE's separate Scratches area.
Files 3 and 5 reuse the prediction function in file 4, so all inputs follow the same steps.

## What changed

The earlier ten office classes have been replaced by these six report classes.
The number at the front of each folder keeps the class IDs in the same order as the report.

| ID | Folder | Meaning |
|---|---|---|
| 0 | `0_aluminium_can` | Aluminium can |
| 1 | `1_plastic_bottle` | Plastic bottle |
| 2 | `2_crumpled_paper` | Crumpled paper |
| 3 | `3_paper_cup` | Paper cup |
| 4 | `4_pen` | Pen |
| 5 | `5_pencil` | Pencil |

There is one model, one training loop and one prediction path. The only random
augmentation is a horizontal flip or small rotation. Learning-rate scheduling,
early stopping, alternative networks and extra command-line settings have been removed.
The best checkpoint is still selected using validation macro-F1.

The Word file is a **dataset and robot design plan**, not a collection of training photos.
This ZIP includes the source document and scanned brief. It contains no real training
images, trained coursework model or final performance results.

## 1 Install

Use Python 3.12. Extract the ZIP before opening it.
Open the `Office_Items_Simple` folder as a project in VS Code or PyCharm.

In a Windows terminal:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Select `.venv\Scripts\python.exe` as the interpreter in your IDE.
Using the full interpreter path avoids PowerShell activation-policy problems.

On macOS or Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The examples below use `python`. In your IDE terminal, activate/select the virtual
environment first, or replace `python` with the full interpreter path above.
The code runs on a CPU; an NVIDIA GPU is not required.

On Linux, you can install the matching CPU-only PyTorch packages first to avoid
downloading GPU packages:

```bash
.venv/bin/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

## 2 Add real photos

Put JPG or PNG photos into the six class folders already supplied under `data/raw`.
Inside each class, use one subfolder per physical item. For example:

```text
data/raw/0_aluminium_can/can_A/photo_01.jpg
data/raw/0_aluminium_can/can_A/photo_02.jpg
data/raw/0_aluminium_can/can_B/photo_01.jpg
data/raw/0_aluminium_can/can_C/photo_01.jpg
```

Keep all photos of can A in `can_A`, including photos taken on another day.
If items are photographed together or share a near-identical background/session,
keep those related images in the same group too. For this classifier, each image
should contain one main target item; crop a separate image for each item if necessary.

You need at least three independent groups per class so every split can contain
the class. That is only a software minimum, not enough evidence of a good model.
Collect substantially more varied items and photos where possible, across lighting,
angles and backgrounds. Do not photograph every class on its own distinctive background.
The report's target of 300-500 labelled instances per class concerns the proposed
detector; it is not a count of images already supplied.

The script aims for 70% train, 15% validation and 15% test **by group**.
Whole groups stay together, so the actual image percentages can differ. The printed
counts and manifest show the real split. Exact byte-identical duplicates are rejected;
near-duplicates still depend on careful grouping and visual checks.

Record where photos came from and any permissions in `data/source_log.csv`.
Fill in `DATASET_CARD.md`. Do not use the three brief scans as training images.

## 3 Run

```bash
python ui.py

```

Training runs for 10 epochs by default. One epoch is one pass through the training
images. The first training run downloads ImageNet-pretrained MobileNet weights;
these are starting weights, not a ready-made six-class coursework model.
Only the new final layer is trained. The feature layers stay frozen.

You can change the number of epochs without editing the file:

```bash
python train_model.py --epochs 15
```

## 4 Read the outputs

| Output | What to look for |
|---|---|
| `data/split/manifest.csv` | Source file, class, group, split and file hash |
| `outputs/best_model.pt` | Weights from the epoch with the highest validation macro-F1 |
| `outputs/training_history.csv` | Training loss, training accuracy, validation accuracy and F1 |
| `outputs/test_results/metrics.json` | Final test accuracy and macro-F1 |
| `outputs/test_results/confusion_matrix.png` | Rows are actual classes; columns are predictions |
| `outputs/test_results/class_report.txt` | Precision, recall and F1 for each class |
| `outputs/test_results/predictions.csv` | Every test image and its prediction |
| `outputs/test_results/error_analysis.md` | Wrong answers to inspect and explain |
| `outputs/prediction.jpg` | The latest single image with its predicted label |

For example, prediction prints `paper cup: <model confidence>`.
No particular class, confidence, accuracy or F1 is promised.
Open the listed wrong photos and add your observations to the generated error
analysis. The software cannot tell whether a mistake was caused by blur or lighting
without you inspecting it.

## Scope and limitations

The scanned Part B task asks for classification of one item at a time. This
prototype answers “Which of these six classes best matches this image?”
It always chooses one of the six classes, even for an unrelated image or empty desk.
Softmax confidence is not a calibrated guarantee of correctness.

The larger robot in the Word report needs a YOLO11n detector, bounding boxes,
depth and robot control. Those remain proposed integration work. This classifier
does not produce positions, mAP values or grasp commands. See `SOURCES.md` for
a short paragraph explaining the relationship in your report.

## Troubleshooting

- **Module not found:** install requirements using the same interpreter selected in the IDE.
- **Missing or empty data:** add real photos in item group folders, then run `prepare_data.py`.
- **Existing split:** keep it for repeatable training. To replace it, move `data/split`
  outside the project first, then prepare again. Never move individual test photos into train.
- **Duplicate photo:** keep one copy, review its group, and run preparation again.
- **Cannot download starting weights:** connect to the internet for the first training run.
  Do not replace transfer learning with random weights and present that as the same experiment.
- **Cannot open camera:** close other camera applications, allow camera access and try camera 1.
- **OpenCV cannot open a window:** run the camera script on your local desktop, not a remote
  headless terminal. Install `opencv-python`, not `opencv-python-headless`.
- **Training is slow:** try `--epochs 2` to check the workflow first. Final training and
  performance assessment still require the real dataset.

