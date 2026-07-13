from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


FIELD_CLASSES = ("id", "birth", "name")
CORNER_CLASSES = ("top_left", "top_right", "bottom_right", "bottom_left")
REQUIRED_CLASSES = FIELD_CLASSES + CORNER_CLASSES

CLASS_ALIASES = {
    "identity_number": "id",
    "identity number": "id",
    "id_number": "id",
    "citizen_id": "id",
    "citizen id": "id",
    "birthday": "birth",
    "date_of_birth": "birth",
    "date of birth": "birth",
    "dob": "birth",
    "full_name": "name",
    "full name": "name",
    "top-left": "top_left",
    "top left": "top_left",
    "topright": "top_right",
    "top-right": "top_right",
    "top right": "top_right",
    "bottomleft": "bottom_left",
    "bottom-left": "bottom_left",
    "bottom left": "bottom_left",
    "bottomright": "bottom_right",
    "bottom-right": "bottom_right",
    "bottom right": "bottom_right",
}


@dataclass(frozen=True)
class Detection:
    name: str
    confidence: float
    xyxy: np.ndarray

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy.tolist()
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class VietOCRReader:
    def __init__(self, device: str = "auto", beamsearch: bool = False) -> None:
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        config = Cfg.load_config_from_name("vgg_transformer")
        config["weights"] = "https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA"
        config["cnn"]["pretrained"] = False
        config["device"] = device
        config["predictor"]["beamsearch"] = beamsearch
        self.predictor = Predictor(config)

    def read(self, crop_bgr: np.ndarray) -> str:
        if crop_bgr.size == 0:
            return ""

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)
        return self.predictor.predict(pil_image).strip()


def normalize_class_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return CLASS_ALIASES.get(normalized, normalized)


def parse_model_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def run_detection(model: YOLO, image_bgr: np.ndarray, conf: float, iou: float) -> list[Detection]:
    result = model.predict(image_bgr, conf=conf, iou=iou, verbose=False)[0]
    model_names = parse_model_names(result.names)
    detections: list[Detection] = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls.item())
        name = normalize_class_name(model_names[class_id])
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.float32)
        confidence = float(box.conf.item())
        detections.append(Detection(name=name, confidence=confidence, xyxy=xyxy))

    return detections


def select_best_by_class(detections: list[Detection]) -> dict[str, Detection]:
    selected: dict[str, Detection] = {}
    for detection in detections:
        if detection.name not in REQUIRED_CLASSES:
            continue
        previous = selected.get(detection.name)
        if previous is None or detection.confidence > previous.confidence:
            selected[detection.name] = detection
    return selected


def perspective_from_corners(
    selected: dict[str, Detection],
    output_width: int,
    output_height: int,
) -> np.ndarray | None:
    if any(corner not in selected for corner in CORNER_CLASSES):
        return None

    source = np.float32([selected[name].center for name in CORNER_CLASSES])
    destination = np.float32(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ]
    )
    return cv2.getPerspectiveTransform(source, destination)


def transform_box(box: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.tolist()
    points = np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])
    transformed = cv2.perspectiveTransform(points, matrix)[0]
    min_xy = transformed.min(axis=0)
    max_xy = transformed.max(axis=0)
    return np.array([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=np.float32)


def crop_box(image_bgr: np.ndarray, box: np.ndarray, padding: float) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = box.tolist()
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad_x = box_width * padding
    pad_y = box_height * padding

    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))

    if right <= left or bottom <= top:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return image_bgr[top:bottom, left:right]


def preprocess_crop(crop_bgr: np.ndarray, min_height: int = 48) -> np.ndarray:
    if crop_bgr.size == 0:
        return crop_bgr

    height, width = crop_bgr.shape[:2]
    if height < min_height:
        scale = min_height / max(1, height)
        crop_bgr = cv2.resize(
            crop_bgr,
            (max(1, int(round(width * scale))), min_height),
            interpolation=cv2.INTER_CUBIC,
        )

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(crop_bgr, -1, sharpen_kernel)


def ocr_field(reader: VietOCRReader, field: str, crop_bgr: np.ndarray) -> str:
    crop_bgr = preprocess_crop(crop_bgr)
    if crop_bgr.size == 0:
        return ""

    candidates = [reader.read(crop_bgr)]
    height, width = crop_bgr.shape[:2]
    if height > 70 and height / max(1, width) > 0.25:
        midpoint = height // 2
        candidates.append(reader.read(crop_bgr[:midpoint, :]))
        candidates.append(reader.read(crop_bgr[midpoint:, :]))

    if field in {"id", "birth"}:
        return max(candidates, key=lambda text: len(re.sub(r"\D", "", text)))
    return max(candidates, key=len)


def clean_id(text: str) -> str:
    digits = re.sub(r"\D", "", text)
    if len(digits) > 12:
        match = re.search(r"\d{12}", digits)
        if match:
            return match.group(0)
    return digits


def clean_birth(text: str) -> str:
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        return digits[:8]
    return digits


def clean_name(text: str) -> str:
    text = text.upper()
    text = "".join(char if char.isalpha() or char.isspace() else " " for char in text)
    text = re.sub(r"\b(HO|HỌ|VA|VÀ|TEN|TÊN|FULL|NAME|NGAY|SINH)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_field(field: str, text: str) -> str:
    if field == "id":
        return clean_id(text)
    if field == "birth":
        return clean_birth(text)
    if field == "name":
        return clean_name(text)
    return text.strip()


def extract_cccd(
    image_path: Path,
    weights_path: Path,
    conf: float,
    iou: float,
    padding: float,
    warp_width: int,
    warp_height: int,
    ocr_device: str,
    beamsearch: bool,
    save_debug: Path | None,
) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    model = YOLO(str(weights_path))
    detections = run_detection(model, image_bgr, conf=conf, iou=iou)
    selected = select_best_by_class(detections)
    warnings: list[str] = []

    missing_fields = [field for field in FIELD_CLASSES if field not in selected]
    if missing_fields:
        warnings.append(f"Missing field detections: {', '.join(missing_fields)}")

    matrix = perspective_from_corners(selected, warp_width, warp_height)
    if matrix is None:
        working_image = image_bgr
        warnings.append("Missing one or more card corners; OCR used original image crops.")
    else:
        working_image = cv2.warpPerspective(image_bgr, matrix, (warp_width, warp_height))

    reader = VietOCRReader(device=ocr_device, beamsearch=beamsearch)
    raw_text: dict[str, str] = {}
    output: dict[str, str] = {}

    for field in FIELD_CLASSES:
        detection = selected.get(field)
        if detection is None:
            raw_text[field] = ""
            output[field] = ""
            continue

        box = transform_box(detection.xyxy, matrix) if matrix is not None else detection.xyxy
        crop = crop_box(working_image, box, padding=padding)
        text = ocr_field(reader, field, crop)
        raw_text[field] = text
        output[field] = clean_field(field, text)

    if save_debug:
        debug = working_image.copy()
        for field in FIELD_CLASSES:
            detection = selected.get(field)
            if detection is None:
                continue
            box = transform_box(detection.xyxy, matrix) if matrix is not None else detection.xyxy
            x1, y1, x2, y2 = box.astype(int).tolist()
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug, field, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        save_debug.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_debug), debug)

    return {
        "id": output["id"],
        "birth": output["birth"],
        "name": output["name"],
        "raw_text": raw_text,
        "warnings": warnings,
        "detections": {
            name: {
                "confidence": round(detection.confidence, 4),
                "xyxy": [round(float(value), 2) for value in detection.xyxy.tolist()],
            }
            for name, detection in selected.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract id, birth, and name from Vietnamese CCCD cards.")
    parser.add_argument("--image", required=True, type=Path, help="Path to a CCCD card image.")
    parser.add_argument("--weights", required=True, type=Path, help="Path to trained YOLOv8 weights.")
    parser.add_argument("--conf", default=0.25, type=float, help="YOLO confidence threshold.")
    parser.add_argument("--iou", default=0.5, type=float, help="YOLO NMS IoU threshold.")
    parser.add_argument("--padding", default=0.05, type=float, help="Padding ratio added around field crops.")
    parser.add_argument("--warp-width", default=856, type=int, help="Warped CCCD image width.")
    parser.add_argument("--warp-height", default=540, type=int, help="Warped CCCD image height.")
    parser.add_argument("--ocr-device", default="auto", help="VietOCR device: auto, cpu, cuda:0, ...")
    parser.add_argument("--beamsearch", action="store_true", help="Enable VietOCR beam search.")
    parser.add_argument("--save-debug", type=Path, help="Optional path to save detected field overlay.")
    args = parser.parse_args()

    result = extract_cccd(
        image_path=args.image,
        weights_path=args.weights,
        conf=args.conf,
        iou=args.iou,
        padding=args.padding,
        warp_width=args.warp_width,
        warp_height=args.warp_height,
        ocr_device=args.ocr_device,
        beamsearch=args.beamsearch,
        save_debug=args.save_debug,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
