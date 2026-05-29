import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_IMAGES_DIR = ROOT / "dataset" / "images"
SRC_LABELS_DIR = SRC_IMAGES_DIR / "DetectLabels"
BASE_DIR = ROOT / "yolo_dataset"

TRAIN_IMAGES_DIR = BASE_DIR / "images" / "train"
VAL_IMAGES_DIR = BASE_DIR / "images" / "val"
TRAIN_LABELS_DIR = BASE_DIR / "labels" / "train"
VAL_LABELS_DIR = BASE_DIR / "labels" / "val"


def main():
    random.seed(42)

    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)

    for directory in (
        TRAIN_IMAGES_DIR,
        VAL_IMAGES_DIR,
        TRAIN_LABELS_DIR,
        VAL_LABELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    labeled_files = []
    for label_path in sorted(SRC_LABELS_DIR.glob("*.txt")):
        if label_path.name == "classes.txt":
            continue

        image_path = SRC_IMAGES_DIR / f"{label_path.stem}.jpg"
        if image_path.exists():
            labeled_files.append((image_path, label_path))

    print(f"Found {len(labeled_files)} labeled images.")

    if not labeled_files:
        print("No labeled images found.")
        return

    random.shuffle(labeled_files)

    split_index = max(1, int(len(labeled_files) * 0.8))
    train_files = labeled_files[:split_index]
    val_files = labeled_files[split_index:]

    for files, image_dir, label_dir in (
        (train_files, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR),
        (val_files, VAL_IMAGES_DIR, VAL_LABELS_DIR),
    ):
        for image_path, label_path in files:
            shutil.copy2(image_path, image_dir / image_path.name)
            shutil.copy2(label_path, label_dir / label_path.name)

    classes_file = SRC_LABELS_DIR / "classes.txt"
    if classes_file.exists():
        classes = [
            line.strip()
            for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        classes = ["head"]

    yaml_lines = [
        f"path: {BASE_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    yaml_lines.extend(f"  {i}: {class_name}" for i, class_name in enumerate(classes))
    (BASE_DIR / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print("Dataset prepared.")
    print(f"Train images: {len(train_files)}")
    print(f"Val images: {len(val_files)}")
    print(f"Classes: {classes}")
    print(f"Data yaml: {BASE_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
