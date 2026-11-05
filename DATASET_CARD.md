# Six class dataset card

## Current status

This package contains a dataset plan and empty class folders. No real training,
validation or test photos were supplied. No coursework accuracy or macro-F1 has
been measured. Complete this card when you collect the photos.

## Purpose and labels

Classify one main office recycling or stationery item per image.
The class IDs come from the target object table in `sources/Restarted_Y3C1_Updated.docx`.

| ID | Label | Raw folder |
|---|---|---|
| 0 | Aluminium can | `0_aluminium_can` |
| 1 | Plastic bottle | `1_plastic_bottle` |
| 2 | Crumpled paper | `2_crumpled_paper` |
| 3 | Paper cup | `3_paper_cup` |
| 4 | Pen | `4_pen` |
| 5 | Pencil | `5_pencil` |

## Collection record to complete

- Collector and collection dates: [complete]
- Camera and locations: [complete]
- Image sources and permission or licence: [complete; use `data/source_log.csv`]
- People or private information visible, and handling: [complete]
- Lighting, surfaces, angles and object variety: [complete]
- Number of independent item groups per class: [complete]
- Number of train, validation and test photos per class: [copy actual preparation counts]
- Excluded or unreadable images and reasons: [complete]
- If a download is needed for submission, the accessible image archive link: [complete]

## Split and preprocessing

Seed 42. Approximately 70/15/15 by independent groups, not necessarily by image
count. At least three groups per class are required by the preparation script.
All views of the same physical item, and other near-identical captures, must stay
in one group. `data/split/manifest.csv` records the exact allocation.
Exact duplicate files are rejected. Group labels and near-duplicates require manual review.

Input is RGB, resized to 224 by 224, converted to a tensor and normalised with
ImageNet means and standard deviations. Training also uses horizontal flipping
and rotation up to 10 degrees. Validation, test and prediction have no random augmentation.

## Limitations and results to complete

A six-way classifier cannot locate items or reliably reject unknown objects.
Background differences, similar pens and pencils, reflections and unseen item
styles may affect generalisation. Review actual mistakes before claiming any causes.

- Final selected checkpoint and settings: [complete]
- Held-out test accuracy and macro-F1: [copy measured values]
- Main observed confusions and examples: [complete]
- Corrective work using training and validation data: [complete]

The report's targets, including 300-500 detector instances per class, are plans.
They are not dataset counts or measured results for this prototype.
