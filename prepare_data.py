"""Sketch 1: check our photos, then make train, val and test folders."""

import csv
import hashlib
import random
import shutil
from pathlib import Path

from PIL import Image

# These numbers match the class IDs in the Word report.
CLASSES = [
    "0_aluminium_can", "1_plastic_bottle", "2_crumpled_paper",
    "3_paper_cup", "4_pen", "5_pencil",
]
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
SPLIT = ROOT / "data" / "split"
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def prepare_data():
    # 1. Stop here if a split already exists. We do not overwrite your work.
    if SPLIT.exists():
        raise ValueError("data/split already exists. Move it elsewhere before making a new split.")
    if not RAW.exists():
        raise ValueError("Create data/raw first. See README.md for the folder names.")
    extra = [p.name for p in RAW.iterdir() if p.is_dir() and p.name not in CLASSES]
    if extra:
        raise ValueError(f"Unrecognised class folders: {extra}. Use the six names in README.md.")

    # 2. Keep photos of the same physical item together in one group folder.
    # Example: data/raw/0_aluminium_can/can_A/photo_01.jpg
    # Do not put the same can into another group, even in a later session.
    rng = random.Random(42)
    seen = {}
    records = []
    for class_name in CLASSES:
        class_folder = RAW / class_name
        if not class_folder.is_dir():
            raise ValueError(f"Missing class folder: {class_folder}")
        if any(p.suffix.lower() in EXTENSIONS for p in class_folder.iterdir()):
            raise ValueError(f"Put the photos inside item group folders under {class_name}.")
        groups = sorted(p for p in class_folder.iterdir() if p.is_dir())
        if len(groups) < 3:
            raise ValueError(f"{class_name} needs at least 3 different item groups.")
        rng.shuffle(groups)

        # About 70/15/15 by groups. Small datasets require rounding.
        n_val = max(1, round(len(groups) * 0.15))
        n_test = max(1, round(len(groups) * 0.15))
        n_train = len(groups) - n_val - n_test
        for index, group in enumerate(groups):
            split = "train" if index < n_train else "val"
            if index >= n_train + n_val:
                split = "test"
            photos = sorted(p for p in group.rglob("*") if p.suffix.lower() in EXTENSIONS)
            if not photos:
                raise ValueError(f"No JPG or PNG photos in {group}")
            for photo in photos:
                # 3. Check that each file is an image before copying anything.
                with Image.open(photo) as image:
                    image.verify()
                digest = hashlib.sha256(photo.read_bytes()).hexdigest()
                if digest in seen:
                    raise ValueError(f"Duplicate photo: {photo} and {seen[digest]}")
                seen[digest] = photo
                relative = photo.relative_to(RAW)
                records.append((relative, class_name, group.name, split, digest))

    # 4. All checks passed. Copy each complete group into its assigned split.
    for relative, class_name, group, split, digest in records:
        destination = SPLIT / split / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RAW / relative, destination)

    # 5. Leave a simple record of where every photo went.
    with (SPLIT / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["raw_file", "class_folder", "item_group", "split", "sha256"])
        writer.writerows(records)
    for class_name in CLASSES:
        counts = [sum(r[1] == class_name and r[3] == s for r in records)
                  for s in ("train", "val", "test")]
        print(f"{class_name}: train={counts[0]}, val={counts[1]}, test={counts[2]}")
    print("Done. Check data/split/manifest.csv, then run train_model.py.")


if __name__ == "__main__":
    try:
        prepare_data()
    except (ValueError, OSError) as error:
        raise SystemExit(f"Cannot prepare data: {error}")
