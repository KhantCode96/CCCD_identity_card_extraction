# YOLOv8 + VietOCR CCCD Extraction

This is the v1 workflow for extracting information from Vietnamese CCCD cards.
It uses a Roboflow YOLOv8 export to detect card corners and three text fields,
then uses pretrained VietOCR to read the cropped field images.

This workflow supports CCCD cards only. Keep raw card images, Roboflow exports,
and trained weights private.

## Install

```bash
pip install -r requirements.txt
```

## Expected Dataset

Download the Roboflow dataset in YOLOv8 format. The export should include a
`data.yaml` file and train/valid/test splits.

Required classes:

```text
id
birth
name
top_left
top_right
bottom_right
bottom_left
```

From inside the `YOLOV8 VIET OCR` folder, validate the export before training:

```bash
python tools/validate_yolov8_dataset.py \
  --data-yaml /content/drive/MyDrive/cccd-training-data/data.yaml
```

## Train in Google Colab

Upload or unzip the Roboflow export to private Google Drive, open this folder
in Colab, then run:

```bash
python tools/train_yolov8_cccd_colab.py \
  --mount-drive \
  --data-yaml /content/drive/MyDrive/cccd-training-data/data.yaml \
  --output-root /content/drive/MyDrive/cccd-training-output \
  --model yolov8s.pt \
  --img-size 640 \
  --epochs 100 \
  --batch 16 \
  --patience 20 \
  --test
```

Expected private outputs:

```text
/content/drive/MyDrive/cccd-training-output/runs/detect/cccd_yolov8/weights/best.pt
/content/drive/MyDrive/cccd-training-output/cccd_yolov8_best.pt
```

## Run Extraction

After copying the trained weight file to a private local location, run from
inside this folder:

```bash
python extract_cccd.py \
  --image ../images/test.jpg \
  --weights cccd_yolov8_best.pt \
  --save-debug outputs/pred_cccd_debug.jpg
```

The script prints JSON:

```json
{
  "id": "006096002049",
  "birth": "20071996",
  "name": "HA SY HUYNH",
  "raw_text": {
    "id": "...",
    "birth": "...",
    "name": "..."
  },
  "warnings": []
}
```

If one or more corners are missing, the extractor falls back to OCR on the
original field boxes and includes a warning.

## Evaluation Checklist

- Record YOLO mAP after training, especially per-class results for `id`,
  `birth`, `name`, and the four corners.
- Test at least 20 held-out CCCD images.
- Manually compare extracted `id`, `birth`, and `name`.
- Check failure cases: blurry image, missing corner, missing field, multiple
  detections for one class, and CPU-only inference.

## Notes

- YOLO detects where the fields are; VietOCR reads the text.
- No manual OCR ground-truth labels are required for v1.
- If VietOCR accuracy is weak, generate field crops with the detector, run
  pretrained OCR, manually correct the output, and fine-tune VietOCR later.
