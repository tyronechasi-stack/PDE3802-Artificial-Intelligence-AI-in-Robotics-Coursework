# Software verification

Checked 26 October 2026 using Python 3.12 on Linux, with CPU inference and training.
The five Python files passed 19 software checks.

## What was exercised

- All five files compile and the three command-line help screens work.
- Preparation processes 126 temporary synthetic fixture images across six classes.
  Whole item groups stay in one split, seed 42 reproduces the manifest, and
  the existing split is protected from accidental overwriting.
- Duplicate, corrupt, ungrouped and incorrectly labelled-folder inputs are rejected.
  Missing class folders and too few independent groups give readable errors.
- The actual ImageNet-pretrained MobileNetV3 Small model completes two training
  epochs, saves a checkpoint and records the training history.
- The feature extractor's parameters and BatchNorm state remain unchanged, while
  the new final layer's weights change during training.
- The saved class IDs match the six report classes. The selected checkpoint has
  the highest validation macro-F1 in the run.
- A single image produces a class/confidence pair and an annotated output image.
  A separate check verifies BGR-to-RGB conversion and the added batch dimension.
- Evaluation produces the metrics, per-class report, predictions, confusion matrix
  and mistake list. Accuracy and macro-F1 are independently recalculated from
  the saved predictions and match the reported values.
- Missing images and missing checkpoints are handled. An invalid epoch count is rejected.
- The webcam loop processes a mocked camera frame and releases resources on normal
  exit and camera-open failure.

The final ZIP is also checked after extraction for file integrity, exactly five
Python files, successful compilation and command-line imports. The four supplied
source files are preserved byte for byte. The four-page Word guide is rendered
and visually reviewed.

## Limits

The synthetic fixtures test whether the software operates. They are not a real
office-item dataset, and their scores are not coursework results. They, their
checkpoint and their output reports are excluded from the deliverable.

A physical webcam was not available. Camera hardware and desktop-window display
must be checked on the user's computer. VS Code, PyCharm, Windows, macOS, OAK-D,
ROS 2, robot control and the full YOLO detector were not exercised here.

No real six-class dataset was supplied, so no final task accuracy, macro-F1 or
physical robot performance is claimed. The first real training run still needs
photos and internet access to download pretrained weights.

## Environment used

| Package | Version |
|---|---|
| PyTorch | 2.14.1+cpu |
| Torchvision | 0.29.1+cpu |
| OpenCV | 5.0.0, distribution opencv-python 5.0.0.93 |
| scikit-learn | 1.8.0 |
| NumPy | 2.5.2 |
| Matplotlib | 3.10.8 |
| Pillow | 12.3.0 |

requirements.txt pins the PyTorch pair and OpenCV distribution. Other libraries
have bounded version ranges; a different platform may select different versions.
