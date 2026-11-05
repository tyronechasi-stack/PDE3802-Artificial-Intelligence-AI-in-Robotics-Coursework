import hashlib
from pathlib import Path

seen = {}
duplicates = []
for photo in sorted(Path("data/raw").rglob("*")):
    if photo.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        digest = hashlib.sha256(photo.read_bytes()).hexdigest()
        if digest in seen:
            duplicates.append((photo, seen[digest]))
        else:
            seen[digest] = photo

if not duplicates:
    raise SystemExit("No duplicates found.")

for photo, original in duplicates:
    print(f"DELETE {photo}\n  keeps {original}")

if input(f"\nDelete {len(duplicates)} duplicate file(s)? Type yes to confirm: ").strip().lower() == "yes":
    for photo, _ in duplicates:
        photo.unlink()
    print("Deleted.")
else:
    print("Nothing deleted.")