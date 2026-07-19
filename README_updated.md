# Vietnamese CCCD Identity Card Information Extraction

[![Prepare dataset in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KhantCode96/CCCD_identity_card_extraction/blob/main/PrepareDataset.ipynb)
[![Train and evaluate in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KhantCode96/CCCD_identity_card_extraction/blob/main/18_7_baseline.ipynb)

An end-to-end pipeline for detecting and extracting information from Vietnamese citizen identity cards (CCCD).

- **YOLOv8s** detects the identity number, date of birth, full name, and four card corners.
- **OpenCV** rectifies the card using a four-point perspective transform.
- **VietOCR** recognizes text in the detected field crops.
- Field-specific post-processing normalizes the extracted values.

The current reproducible workflow is split into two notebooks:

1. [`PrepareDataset.ipynb`](PrepareDataset.ipynb) downloads, converts, and validates the dataset.
2. [`18_7_baseline.ipynb`](18_7_baseline.ipynb) trains YOLOv8s, evaluates the best checkpoint, and tests the complete extraction pipeline.

> This repository supports Vietnamese CCCD cards only. Do not commit Roboflow credentials, raw identity-card images, private datasets, debug images containing personal information, or trained weights without confirming that they may legally be shared.

## Pipeline

```text
Roboflow dataset (YOLOv8 export)
        |
        v
Convert polygon/OBB labels to YOLO detection boxes
        |
        v
Validate classes, splits, images, labels, and coordinates
        |
        v
Train and evaluate YOLOv8s
        |
        v
Detect: id, birth, name, and four card corners
        |
        v
Perspective correction using the detected corners
        |
        v
Crop and preprocess the three text fields
        |
        v
VietOCR recognition and field-specific cleaning
        |
        v
JSON result, warnings, and optional debug image
```

## Current Dataset

The executed dataset-preparation notebook uses:

```text
Roboflow workspace: ocrtestingdataset
Roboflow project:   cccd_dataset-xz2sc
Dataset version:    8
Export format:      yolov8
```

The detector uses seven classes:

| ID | Class |
|---:|---|
| 0 | `birth` |
| 1 | `bottom_left` |
| 2 | `bottom_right` |
| 3 | `id` |
| 4 | `name` |
| 5 | `top_left` |
| 6 | `top_right` |

Validated dataset statistics:

| Split | Images | Annotations |
|---|---:|---:|
| Train | 2,660 | 17,275 |
| Validation | 550 | 3,493 |
| Test | 63 | 408 |
| **Total** | **3,273** | **21,176** |

Dataset preparation processed 3,273 label files and converted 114 polygon/OBB rows into standard five-value YOLO detection rows. The final validation found zero invalid rows.

## Repository Structure

```text
.
├── PrepareDataset.ipynb              # Download, convert, and validate data
├── 18_7_baseline.ipynb               # Train, evaluate, and test extraction
├── extract_cccd.py                    # End-to-end detection and OCR CLI
├── requirements.txt
└── tools/
    ├── train_yolov8_cccd_colab.py    # Training and checkpoint export
    └── validate_yolov8_dataset.py    # Dataset integrity checks
```

## Installation

```bash
git clone https://github.com/KhantCode96/CCCD_identity_card_extraction.git
cd CCCD_identity_card_extraction
pip install -r requirements.txt
```

The notebooks are designed for Google Colab with Google Drive mounted at:

```text
/content/drive/MyDrive/YOLOV8-VIET_OCR
```

If you use a different directory, update the paths in both notebooks.

## 1. Prepare the Dataset

Open [`PrepareDataset.ipynb`](PrepareDataset.ipynb) with the first Colab badge.

Before running it:

1. Add `ROBOFLOW_API_KEY` to Colab Secrets.
2. Allow the notebook to access that secret.
3. Mount Google Drive.
4. Confirm the project directory and dataset output paths.

The notebook then:

1. installs the requirements;
2. downloads Roboflow dataset version 8;
3. converts polygon or oriented-box annotations to axis-aligned YOLO boxes;
4. saves the original labels in `labels_polygon_backup` directories;
5. verifies that all converted rows use `class x_center y_center width height`;
6. runs `tools/validate_yolov8_dataset.py`.

The prepared dataset is stored at:

```text
/content/drive/MyDrive/YOLOV8-VIET_OCR/CCCD_dataset-8
```

You can rerun validation manually:

```bash
python tools/validate_yolov8_dataset.py \
  --data-yaml /content/drive/MyDrive/YOLOV8-VIET_OCR/CCCD_dataset-8/data.yaml
```

A successful validation ends with `Dataset validation passed.`

## 2. Train the Baseline Detector

Open [`18_7_baseline.ipynb`](18_7_baseline.ipynb) with the second Colab badge. The completed baseline used a Tesla T4 and the following command:

```bash
python tools/train_yolov8_cccd_colab.py \
  --project-root /content/drive/MyDrive/YOLOV8-VIET_OCR \
  --data-yaml /content/drive/MyDrive/YOLOV8-VIET_OCR/CCCD_dataset-8/data.yaml \
  --output-root /content/drive/MyDrive/YOLOV8-VIET_OCR/training_output/fresh_run \
  --model yolov8s.pt \
  --name fresh_yolov8s \
  --img-size 640 \
  --epochs 100 \
  --batch 16 \
  --patience 20 \
  --device 0 \
  --skip-install
```

The script performs an import smoke test, validates the dataset, trains the detector, and copies the best checkpoint to:

```text
/content/drive/MyDrive/YOLOV8-VIET_OCR/training_output/fresh_run/cccd_yolov8_best.pt
```

## Test-Set Results

The best checkpoint was evaluated on the 63-image test split at an image size of 640 and batch size of 16.

| Metric | Score |
|---|---:|
| Precision | 0.880 |
| Recall | 0.884 |
| mAP50 | 0.893 |
| mAP75 | 0.570 |
| mAP50-95 | 0.569 |

Per-class results:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| `birth` | 0.969 | 0.984 | 0.990 | 0.864 |
| `bottom_left` | 0.819 | 0.875 | 0.854 | 0.413 |
| `bottom_right` | 0.844 | 0.860 | 0.803 | 0.319 |
| `id` | 0.969 | 0.984 | 0.994 | 0.860 |
| `name` | 0.987 | 0.984 | 0.985 | 0.855 |
| `top_left` | 0.802 | 0.836 | 0.859 | 0.385 |
| `top_right` | 0.772 | 0.667 | 0.767 | 0.290 |

These are detector metrics, not OCR exact-match accuracy. The test set is also small, so the results should be interpreted as baseline evidence rather than a final estimate of real-world performance. Text-field detection is substantially stronger than precise corner localization.

## 3. Run End-to-End Extraction

Download the pretrained VietOCR `vgg_transformer.pth` file or run the download cell in `18_7_baseline.ipynb`. The executed notebook stores it under:

```text
/content/drive/MyDrive/YOLOV8-VIET_OCR/weights/vietocr/vgg_transformer.pth
```

Run extraction on one CCCD image:

```bash
python extract_cccd.py \
  --image path/to/test_cccd.jpg \
  --weights /content/drive/MyDrive/YOLOV8-VIET_OCR/training_output/fresh_run/cccd_yolov8_best.pt \
  --save-debug outputs/test_cccd_debug.jpg
```

For each image, `extract_cccd.py`:

1. runs YOLOv8 detection;
2. retains the highest-confidence detection for each required class;
3. applies a four-point perspective transform when all four corners are present;
4. falls back to crops from the original image and adds a warning if corners are missing;
5. crops and preprocesses the `id`, `birth`, and `name` fields;
6. recognizes the crops using VietOCR `vgg_transformer`;
7. cleans each field and returns structured JSON;
8. optionally saves a debug overlay.

Example output from the executed notebook:

```json
{
  "id": "042168010024",
  "birth": "09091968",
  "name": "NGUYỄN THỊ TUYẾN",
  "raw_text": {
    "id": "042168010024",
    "birth": "09/09/1968",
    "name": "NGUYỄN THỊ TUYẾN"
  },
  "warnings": []
}
```

The notebook also runs a reproducible smoke test on 15 randomly selected test images (`random.seed(42)`), saves one JSON result and debug image per input, and writes a `batch_results.csv` report. All 15 commands completed successfully; however, this only confirms pipeline execution. Some recognized values were visibly incorrect, so OCR accuracy still requires ground-truth evaluation.

## Current Status

- [x] Download Roboflow dataset version 8
- [x] Convert polygon/OBB labels to YOLO detection boxes
- [x] Validate dataset structure and annotations
- [x] Train the YOLOv8s baseline
- [x] Evaluate the best checkpoint on the test split
- [x] Run single-image end-to-end extraction
- [x] Run a 15-image extraction smoke test
- [ ] Build field-level OCR ground truth
- [ ] Measure ID, birth-date, and name exact-match accuracy and character error rate
- [ ] Improve corner localization and evaluate perspective correction separately
- [ ] Test on external, real-world CCCD images outside the Roboflow split

## Privacy and Security

- Store `ROBOFLOW_API_KEY` only in Colab Secrets or environment variables.
- Keep raw CCCD images, annotations, model weights, and generated debug images private unless sharing is explicitly permitted.
- Remove or redact personal information from screenshots, logs, JSON examples, and experiment reports before publishing.
- Review repository history as well as the current files before making a privacy-sensitive repository public.

## Limitations

- The detector was evaluated on only 63 test images from the same dataset source.
- Corner classes have lower mAP50-95 than the three text-field classes.
- Successful execution does not guarantee correct OCR output.
- Missing corners cause the extractor to use original-image crops, which may reduce OCR quality under rotation or perspective distortion.
- Performance on unseen cameras, glare, blur, occlusion, and different CCCD layouts has not yet been established.
