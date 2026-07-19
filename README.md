# Vietnamese CCCD Identity Card Information Extraction

[![Open `17_7_run.ipynb` in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KhantCode96/CCCD_identity_card_extraction/blob/main/17_7_run.ipynb)

An end-to-end pipeline for extracting information from Vietnamese citizen identity cards (CCCD) using:

- **YOLOv8** to detect the identity number, date of birth, full name, and four card corners.
- **OpenCV** to rectify the card with a perspective transform.
- **VietOCR** to recognize text from the detected field crops.
- Rule-based post-processing to normalize the final `id`, `birth`, and `name` values.

The latest experiment is documented in [`17_7_run.ipynb`](17_7_run.ipynb). The notebook currently covers Colab setup, Roboflow dataset download, annotation conversion, and dataset validation. Training, evaluation, and end-to-end extraction are provided by the repository scripts.

> This repository supports Vietnamese CCCD cards only. Do not commit raw card images, Roboflow credentials, private datasets, or trained weights.

## Current Pipeline

```text
CCCD image
   |
   v
YOLOv8 detection
   |-- text fields: id, birth, name
   `-- card corners: top_left, top_right, bottom_right, bottom_left
   |
   v
Perspective correction with the four detected corners
   |
   v
Crop and preprocess the three text fields
   |
   v
VietOCR recognition
   |
   v
Field-specific text cleaning
   |
   v
JSON result + warnings + optional debug image
```

### Dataset and Training Workflow

1. Mount Google Drive in Colab.
2. Install the project dependencies.
3. load `ROBOFLOW_API_KEY` from Colab Secrets.
4. Download Roboflow project version 8 in YOLOv8 format.
5. Convert polygon or oriented-box annotations into standard YOLO detection boxes.
6. Back up the original labels as `labels_polygon_backup`.
7. Validate class names, split paths, annotation rows, and normalized coordinates.
8. Train a YOLOv8 detector.
9. Evaluate the best checkpoint on the validation split and optionally the test split.
10. Run `extract_cccd.py` for perspective correction, OCR, and JSON output.

## Current Dataset Snapshot

The executed `17_7_run.ipynb` notebook uses:

```text
Roboflow workspace: ocrtestingdataset
Roboflow project:   cccd_dataset-xz2sc
Dataset version:    8
Export format:      yolov8
```

Validated classes:

```text
birth
bottom_left
bottom_right
id
name
top_left
top_right
```

Validation results from the current run:

| Split | Images | Annotations |
|---|---:|---:|
| Train | 2,660 | 17,275 |
| Validation | 550 | 3,493 |
| Test | 63 | 408 |
| **Total** | **3,273** | **21,176** |

The conversion stage changed **114 polygon/OBB rows** into standard five-value YOLO detection rows. The subsequent format check reported **0 invalid rows**.

## Repository Structure

```text
.
├── 17_7_run.ipynb
├── extract_cccd.py
├── requirements.txt
└── tools/
    ├── train_yolov8_cccd_colab.py
    └── validate_yolov8_dataset.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/KhantCode96/CCCD_identity_card_extraction.git
cd CCCD_identity_card_extraction
```

Install the dependencies:

```bash
pip install -r requirements.txt
pip install roboflow
```

`roboflow` is installed separately because the current notebook uses it to download the dataset, but it is not yet listed in `requirements.txt`.

## Run the Current Colab Notebook

Open [`17_7_run.ipynb`](17_7_run.ipynb) using the Colab badge at the top of this README.

Before running the notebook:

1. Add a Colab secret named `ROBOFLOW_API_KEY`.
2. Give the notebook permission to access the secret.
3. Mount Google Drive.
4. Update the working directory if your project is not stored at:

```text
/content/drive/MyDrive/YOLOV8-VIET_OCR
```

The current notebook downloads the dataset to:

```text
/content/drive/MyDrive/YOLOV8-VIET_OCR/CCCD_dataset-8
```

Keep the API key in Colab Secrets. Never paste it directly into the notebook or commit it to GitHub.

## Annotation Conversion

Roboflow exports may contain a mixture of:

```text
class_id x_center y_center width height
```

and polygon or oriented-box rows:

```text
class_id x1 y1 x2 y2 ... xn yn
```

The conversion cell in `17_7_run.ipynb`:

- keeps valid five-value YOLO rows unchanged;
- converts polygon/OBB coordinates to an axis-aligned bounding box;
- checks that all coordinates are numeric and normalized to `0..1`;
- rejects unsupported or zero-area annotations;
- validates all rows before modifying any label file;
- creates a `labels_polygon_backup` directory for each available split.

## Validate the Dataset

Run the validator before training:

```bash
python tools/validate_yolov8_dataset.py \
  --data-yaml /content/drive/MyDrive/YOLOV8-VIET_OCR/CCCD_dataset-8/data.yaml
```

The validator checks:

- the required seven classes;
- train, validation, and test paths;
- matching image and label files;
- five-value YOLO detection rows;
- valid class indices;
- coordinates normalized to `0..1`;
- per-class annotation counts.

A successful run ends with:

```text
Dataset validation passed.
```

## Train YOLOv8 in Google Colab

After validation, train the detector with:

```bash
%cd "/content/drive/MyDrive/YOLOV8-VIET_OCR"

!python tools/train_yolov8_cccd_colab.py \
  --project-root "/content/drive/MyDrive/YOLOV8-VIET_OCR" \
  --data-yaml "/content/drive/MyDrive/YOLOV8-VIET_OCR/CCCD_dataset-8/data.yaml" \
  --output-root "/content/drive/MyDrive/YOLOV8-VIET_OCR/training_output/18_7_baseline" \
  --model yolov8s.pt \
  --name 18_7_yolov8s_baseline \
  --img-size 640 \
  --epochs 100 \
  --batch 16 \
  --patience 20 \
  --device 0 \
  --skip-install
```

The training script:

1. installs the main runtime dependencies unless `--skip-install` is used;
2. performs an import smoke test;
3. validates the dataset;
4. trains the YOLOv8 detector;
5. copies the best checkpoint to a stable output path;
6. evaluates the best model on the validation split;
7. evaluates the test split when `--test` is supplied.

Expected private outputs:

```text
/content/drive/MyDrive/cccd-training-output/runs/detect/cccd_yolov8/weights/best.pt
/content/drive/MyDrive/cccd-training-output/cccd_yolov8_best.pt
```

## Run End-to-End Extraction

Run the extractor with a trained YOLOv8 checkpoint:

```bash
python extract_cccd.py \
  --image path/to/test_cccd.jpg \
  --weights /content/drive/MyDrive/cccd-training-output/cccd_yolov8_best.pt \
  --save-debug outputs/test_cccd_debug.jpg
```

Useful optional arguments:

```text
--conf 0.25
--iou 0.5
--padding 0.05
--warp-width 856
--warp-height 540
--ocr-device auto
--beamsearch
```

### Extraction Logic

For each image, `extract_cccd.py`:

1. runs YOLOv8 detection;
2. keeps the highest-confidence detection for each required class;
3. creates a perspective transform when all four corners are present;
4. warps the card to `856 x 540` by default;
5. transforms the field boxes into the rectified coordinate system;
6. pads and crops the `id`, `birth`, and `name` regions;
7. upscales small crops and applies sharpening;
8. runs pretrained VietOCR using `vgg_transformer`;
9. cleans each field using field-specific rules;
10. returns structured JSON and optionally saves a debug overlay.

When one or more corners are missing, the extractor falls back to the original image and adds a warning instead of stopping.

## Example Output

```json
{
  "id": "006096002049",
  "birth": "20071996",
  "name": "HA SY HUYNH",
  "raw_text": {
    "id": "006096002049",
    "birth": "20/07/1996",
    "name": "HÀ SỸ HUỲNH"
  },
  "warnings": [],
  "detections": {
    "id": {
      "confidence": 0.97,
      "xyxy": [120.5, 245.8, 510.2, 302.4]
    }
  }
}
```

## Evaluation Checklist

### Detector

- Record overall and per-class precision, recall, mAP50, and mAP50-95.
- Inspect performance separately for the three text fields and four corners.
- Review false positives, missed corners, and duplicate field detections.
- Test images with rotation, perspective distortion, glare, blur, and partial occlusion.

### OCR and End-to-End Extraction

- Prepare a held-out set with ground-truth values for `id`, `birth`, and `name`.
- Measure exact-match accuracy for each field.
- Measure character error rate for OCR outputs.
- Compare OCR before and after perspective correction.
- Review failures caused by poor crops, low resolution, glare, and unusual fonts.
- Test both GPU and CPU inference.

## Current Status and Next Step

The current `17_7_run.ipynb` execution completes dataset preparation and validation successfully. The next pipeline step is to run `tools/train_yolov8_cccd_colab.py`, save the best checkpoint, record validation/test metrics, and then evaluate `extract_cccd.py` on held-out CCCD images with ground-truth text.

## Privacy and Security

- Store `ROBOFLOW_API_KEY` only in Colab Secrets or environment variables.
- Keep raw CCCD images and annotations outside the public repository.
- Keep trained weights private unless the dataset and model can legally be shared.
- Remove personal information from screenshots, logs, and debug outputs before publishing.
