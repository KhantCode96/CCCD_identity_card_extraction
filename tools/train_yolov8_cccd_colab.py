"""
Colab-first YOLOv8 training entrypoint for CCCD field detection.

Example:
    python tools/train_yolov8_cccd_colab.py \
      --mount-drive \
      --data-yaml /content/drive/MyDrive/cccd_dataset/data.yaml \
      --output-root /content/drive/MyDrive/cccd-training-output
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path | None = None) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def pip_install(packages: list[str]) -> None:
    run([sys.executable, "-m", "pip", "install", *packages])


def ensure_colab_drive_mounted() -> None:
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        print("Not running inside Google Colab; skipping Drive mount.")
        return

    drive.mount("/content/drive")


def install_dependencies(skip_install: bool) -> None:
    if skip_install:
        return
    pip_install(["ultralytics>=8.2", "vietocr", "opencv-python-headless>=4.7", "PyYAML"])


def validate_dataset(project_root: Path, data_yaml: Path, skip_validation: bool) -> None:
    if skip_validation:
        return
    validator = project_root / "tools" / "validate_yolov8_dataset.py"
    if not validator.exists():
        raise FileNotFoundError(f"Dataset validator was not found: {validator}")
    run([sys.executable, str(validator), "--data-yaml", str(data_yaml)])


def train(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    model = YOLO(args.model)
    train_args = {
        "data": str(args.data_yaml),
        "imgsz": args.img_size,
        "epochs": args.epochs,
        "batch": args.batch,
        "patience": args.patience,
        "project": str(args.output_root / "runs" / "detect"),
        "name": args.name,
        "exist_ok": True,
    }
    if args.device is not None:
        train_args["device"] = args.device
    model.train(**train_args)

    best = args.output_root / "runs" / "detect" / args.name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"YOLOv8 best checkpoint was not created: {best}")

    exported = args.output_root / "cccd_yolov8_best.pt"
    shutil.copy2(best, exported)
    print(f"Exported best checkpoint to {exported}")
    return best


def evaluate(args: argparse.Namespace, best: Path) -> None:
    from ultralytics import YOLO

    model = YOLO(str(best))
    val_args = {"data": str(args.data_yaml), "imgsz": args.img_size}
    if args.device is not None:
        val_args["device"] = args.device
    model.val(**val_args, split="val")
    if args.test:
        model.val(**val_args, split="test")


def smoke_test_imports() -> None:
    import cv2  # noqa: F401
    import torch  # noqa: F401
    import ultralytics  # noqa: F401
    import vietocr  # noqa: F401

    print("Import smoke test passed: cv2, torch, ultralytics, vietocr")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 CCCD detector in Colab.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-yaml", required=True, type=Path, help="Path to Roboflow YOLOv8 data.yaml.")
    parser.add_argument("--output-root", required=True, type=Path, help="Private output folder for runs and weights.")
    parser.add_argument("--model", default="yolov8s.pt", help="Ultralytics base model.")
    parser.add_argument("--name", default="cccd_yolov8", help="Run name under runs/detect.")
    parser.add_argument("--img-size", default=640, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0, cpu. Default lets Ultralytics choose.")
    parser.add_argument("--mount-drive", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--test", action="store_true", help="Also evaluate the test split after validation.")
    args = parser.parse_args()

    args.project_root = args.project_root.resolve()
    args.data_yaml = args.data_yaml.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.mount_drive:
        ensure_colab_drive_mounted()

    install_dependencies(args.skip_install)
    smoke_test_imports()
    validate_dataset(args.project_root, args.data_yaml, args.skip_validation)
    best = train(args)
    evaluate(args, best)
    print("Done.")


if __name__ == "__main__":
    main()
