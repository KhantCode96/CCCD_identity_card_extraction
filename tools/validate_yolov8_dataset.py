"""
Validate a Roboflow/Ultralytics YOLOv8 dataset export for this folder.

The expected layout is the standard YOLOv8 export with a data.yaml file and
train/valid/test image-label splits. The validator checks paths, class names,
YOLO label rows, normalized coordinates, and per-class counts without exposing
or copying private card images.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


REQUIRED_CLASSES = {"id", "birth", "name", "top_left", "top_right", "bottom_right", "bottom_left"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def load_data_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"data.yaml not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        fail(f"Invalid YAML content: {path}")
    return data


def parse_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names")
    if isinstance(names, dict):
        ordered = [names[index] for index in sorted(names)]
    elif isinstance(names, list):
        ordered = names
    else:
        fail("data.yaml must contain names as a list or dict")

    parsed = [normalize_name(str(name)) for name in ordered]
    missing = REQUIRED_CLASSES - set(parsed)
    if missing:
        fail(f"Dataset is missing required classes: {sorted(missing)}")
    return parsed


def resolve_split_path(data_yaml: Path, data: dict[str, Any], raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if "path" in data:
        root = Path(data["path"])
        if not root.is_absolute():
            root = data_yaml.parent / root
        return (root / path).resolve()
    return (data_yaml.parent / path).resolve()


def split_label_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts)
    return images_dir.parent / "labels"


def image_files(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def validate_split(data_yaml: Path, data: dict[str, Any], split_key: str, names: list[str]) -> Counter[str]:
    if split_key not in data:
        return Counter()

    images_dir = resolve_split_path(data_yaml, data, data[split_key])
    labels_dir = split_label_dir(images_dir)
    if not images_dir.exists():
        fail(f"Missing {split_key} image directory: {images_dir}")
    if not labels_dir.exists():
        fail(f"Missing {split_key} label directory: {labels_dir}")

    images = image_files(images_dir)
    if not images:
        fail(f"No images found in {images_dir}")

    counts: Counter[str] = Counter()
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            fail(f"Missing label file for {image_path.name}: {label_path}")

        rows = [row.strip() for row in label_path.read_text(encoding="utf-8").splitlines() if row.strip()]
        for row in rows:
            parts = row.split()
            if len(parts) != 5:
                fail(f"Invalid YOLO row in {label_path}: {row}")
            try:
                class_id = int(float(parts[0]))
                values = [float(value) for value in parts[1:]]
            except ValueError:
                fail(f"Non-numeric YOLO row in {label_path}: {row}")

            if class_id < 0 or class_id >= len(names):
                fail(f"Class id out of range in {label_path}: {class_id}")
            if any(value < 0.0 or value > 1.0 for value in values):
                fail(f"YOLO coordinates must be normalized 0..1 in {label_path}: {row}")

            counts[names[class_id]] += 1

    print(f"{split_key}: images={len(images)} labels={sum(counts.values())}")
    for class_name in names:
        print(f"  {class_name}: {counts[class_name]}")

    missing_counts = [class_name for class_name in REQUIRED_CLASSES if counts[class_name] == 0]
    if missing_counts:
        fail(f"{split_key} split has no labels for required classes: {missing_counts}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a YOLOv8 CCCD dataset export.")
    parser.add_argument("--data-yaml", required=True, type=Path, help="Path to Roboflow YOLOv8 data.yaml.")
    args = parser.parse_args()

    data_yaml = args.data_yaml.resolve()
    data = load_data_yaml(data_yaml)
    names = parse_names(data)
    print(f"Classes: {names}")

    total = Counter()
    for split_key in ("train", "val", "valid", "test"):
        total.update(validate_split(data_yaml, data, split_key, names))

    if not total:
        fail("No train/val/valid/test splits were found in data.yaml")

    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
