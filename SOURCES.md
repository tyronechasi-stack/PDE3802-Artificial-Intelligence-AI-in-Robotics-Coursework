# Sources and changes

## Supplied project sources

1. `sources/Restarted_Y3C1_Updated.docx` — supplied robot proposal. Section 5 and
   its target object table define the six new classes and fixed IDs. The proposal
   also describes grouped splitting, YOLO detection and future depth integration.
   The source file is preserved unchanged.
2. `sources/Scan_20260923_102451.jpg` — photographed Part B brief. Requires 6-10
   classes; file or standard-camera input; one item at a time; predicted class
   and confidence where available; training/validation/test images; dataset card;
   README; Git; accuracy; macro-F1; confusion matrix; brief error analysis; and
   a two-minute recorded code walkthrough.
3. `sources/Scan_20260923_101258.jpg` and `sources/Scan_20260923_102017.jpg` —
   photographed Part A report requirements and continuation into Part B.

The supplied Word file describes a dataset; it does not embed an object-photo
dataset or link to a downloadable one. Its embedded picture is a robot software
architecture diagram. The three scans show the assignment, not examples of the
six object classes. They must not be treated as training photos.

## Changes to the earlier code design

- Replaced mug, bottle, book, notebook, stapler, mouse, phone, chair, bin and keyboard
  with the six classes in the current Word report.
- Used numbered folder names so ImageFolder's alphabetical order matches report IDs 0-5.
- Reduced the implementation to five commented Python files.
- Kept MobileNetV3 Small transfer learning for the one-item Part B classifier.
  Only its new final classification layer is trained.
- Kept image and webcam input, class/confidence, held-out evaluation and a confusion matrix.
- Kept related photographs together by item group before splitting.
- Added an image source log and a dataset card template. No public image dataset was downloaded.
- Removed the test-only CNN, scheduler, early stopping and other optional branches.

## Suggested clarification for the robot report

The Part B prototype classifies one main item in an image into the same six
classes used by the robot proposal. It uses a pretrained MobileNetV3 Small
feature extractor and trains a new six-class final layer. It is evaluated using
held-out classification accuracy, macro-F1 and a confusion matrix. The complete
robot proposed in Part A requires a separate YOLO11n object detector to obtain
bounding boxes for depth-based localisation. Detector training, 3D localisation,
navigation and grasping remain future integration work. Classification results
from this prototype must not be presented as detector mAP or robot performance.

## Technical references

These are library and model documentation, not training-image sources.
Checked 26 October 2026.

- PyTorch, MobileNetV3 Small API and pretrained weights:
  https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html
- PyTorch, transfer learning tutorial:
  https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- scikit-learn, F1 and macro averaging:
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- Ultralytics, separate YOLO11 detection and classification variants:
  https://docs.ultralytics.com/models/yolo11/

ImageNet-pretrained model weights are downloaded by Torchvision on the first
training run. No ImageNet image files are included or used as the custom held-out
coursework test set. Report the pretrained starting point in your methodology.
