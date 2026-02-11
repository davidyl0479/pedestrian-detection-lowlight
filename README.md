# Pedestrian Detection in Low-Light (NightOwls + YOLO)

End-to-end computer vision project for low-light pedestrian detection using NightOwls and YOLO (v11s for reported experiments), built during my MSc dissertation and now maintained as a portfolio project.

## Portfolio Snapshot

- Built a reproducible training/evaluation pipeline for NightOwls pedestrian detection.
- Implemented pixel-space COCO prediction export to avoid coordinate-space evaluation errors.
- Added NightOwls-focused MR-2 evaluation alongside COCO AP metrics.
- Structured experiments for fair baseline vs enhancement comparisons (for example Zero-DCE).

## Key Results

### Core model outcomes (YOLOv11s, 640 px, 60 epochs)

| Stage | mAP50-95 | AP50 | Precision | Recall | Notes |
|---|---:|---:|---:|---:|---|
| Training (best epoch 60) | 72.23% | 97.13% | 93.63% | 95.14% | Converged smoothly; best weights used for downstream eval |
| Validation | 52.11% | 71.97% | — | — | AP75: 61.16%, AR100: 56.15% |
| Test | 14.41% | 27.97% | — | — | AP75: 12.43%, AR100: 23.82% |

### Pedestrian-specific and stratified highlights

- Pedestrian AP50-95: **69.41% (validation)** vs **39.95% (test)**.
- Strong size effect on test pedestrians: **small 15.97%**, **medium 41.47%**, **large 55.36%** AP50-95.
- Stratified analysis showed best bins around mid-brightness + lower noise (e.g., AP_R50 up to **62.58%**, MR-2 as low as **3.50%**) and worst bins under darker/noisier conditions (MR-2 up to **~46%**).

> These results come from dissertation evaluation settings: confidence=0.001, IoU=0.50, max_det=300, agnostic_nms=false.

## Engineering Decisions & Tradeoffs

- **Pixel-space COCO export over raw letterbox eval outputs**
  - Ensures coordinates used for COCO scoring match original image space.
- **Fixed evaluation knobs across experiments**
  - `imgsz/conf/iou/max_det` are kept constant to make baseline vs enhancement comparison fair.
- **COCO AP + MR-2 together**
  - AP gives general detector quality, MR-2 captures NightOwls pedestrian miss-rate behavior.
- **Reproducibility-first run structure**
  - Runner stores run artifacts and metadata to support repeatable experiments.

## Skills Demonstrated

- Computer vision model development (YOLOv11s/YOLO pipeline, low-light detection context)
- Data engineering for detection pipelines (COCO → YOLO labels/splits)
- Metric rigor and evaluation design (COCO AP + MR-2)
- Reproducible ML experimentation and artifact management
- Python scripting/automation for end-to-end workflows

## Architecture (high-level)

```text
NightOwls COCO JSON + images
          │
          ▼
prepare_nightowls.py  →  YOLO labels + split indexes
          │
          ▼
YOLO training (baseline, YOLOv11s reported)
          │
          ├── Ultralytics val/test (run folder + base artifacts)
          │
          └── predict_to_coco.py (pixel-space predictions.json)
                         │
                         ▼
run_yolo_baseline.py report  → metrics_by_class_area.csv + pr_data.npz
                         │
                         ▼
missrate_nightowls.py   → MR-2
```

## Quick Demo (fast path)

```bash
# 1) Train (or use existing best.pt)
yolo detect train data=configs/data.yaml model=yolo11s.pt imgsz=640 epochs=60 device=0

# 2) Create eval folder
python scripts/yolo_runs/run_yolo_baseline.py test \
  --data configs/data.yaml \
  --model runs/train/<exp>/weights/best.pt \
  --imgsz 640 --conf 0.001 --iou 0.50 --max_det 300 --workers 8

# 3) Export pixel-space detections
python scripts/yolo_runs/predict_to_coco.py \
  --data configs/data.yaml \
  --gt data/raw/annotations/val/nightowls_validation.json \
  --model runs/train/<exp>/weights/best.pt \
  --split test \
  --imgsz 640 --conf 0.001 --iou 0.50 --max_det 300 \
  --outdir runs/predict/baseline_test

# 4) Compute MR-2
python scripts/metrics/missrate_nightowls.py \
  --gt data/raw/annotations/val/nightowls_validation.json \
  --pred runs/predict/baseline_test/predictions.json \
  --iou 0.5 --min_height 50
```

---

## Technical Reference (trimmed)

### Minimal repo map

- `scripts/prepare_data/`: label/index generation
- `scripts/yolo_runs/`: training/eval/report tooling
- `scripts/metrics/`: MR-2 evaluator
- `scripts/enhance/`: enhancement dataset tools
- `configs/`: dataset/training YAMLs
- `pipeline.md`: canonical detailed runbook
- `QUICKSTART.md`: short onboarding guide

### Requirements (minimal)

```bash
pip install ultralytics pycocotools torch pandas pyyaml pillow tqdm
```

### Common outputs

- `runs/train/<exp>/weights/best.pt`
- `runs/predict/<name>/predictions.json`
- eval folder artifacts: `metrics_by_class_area.csv`, `pr_data.npz`

### Common pitfalls

- All-zero AP: usually wrong coordinate space or wrong `predictions.json` in eval folder.
- Worker/file-handle issues (Windows): lower `--workers`.
- Split mismatch: ensure `--gt` and `--split` refer to the same split.

### Related docs

- `pipeline.md` (full sequence)
- `QUICKSTART.md` (compact setup)
