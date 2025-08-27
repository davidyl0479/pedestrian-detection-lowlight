# 0) Setup (once per run)

```powershell
# (optional) pick your latest trained weight
$best = (Get-ChildItem runs\train\*\weights\best.pt |
         Sort-Object LastWriteTime -Desc | Select -First 1).FullName
```

---

# 1) Train

Resume training from the last checkpoint (Ultralytics):

```powershell
# Option A (explicit path)
yolo detect train resume=True model="runs\train\<your-exp>\weights\last.pt" data=configs\data.yaml imgsz=640 device=0

# Option B (same project/name auto-resume)
yolo detect train resume=True project=runs/train name=<your-exp>
```
If NEW TRAINING, then: 

Use whichever you normally use; both end up with `runs/train/<exp>/weights/best.pt`.

**Option A — Ultralytics CLI**

```powershell
yolo detect train data=configs\data.yaml model=yolov8n.pt imgsz=640 epochs=... device=0
```

**Option B — your runner (if you have a `train` subcmd)**

```powershell
python .\scripts\yolo_runs\run_yolo_baseline.py train --data configs\data.yaml --imgsz 640 --workers 8
```

▶️ **Produces**

* `runs\train\<exp>\weights\best.pt` (and `last.pt`) — use `best.pt` in the next steps.

---

# 2) Create an eval run folder (Ultralytics val/test)

This runs Ultralytics’ internal evaluation (for the console header) **and** creates the eval directory we’ll reuse.

```powershell
python .\scripts\yolo_runs\run_yolo_baseline.py test `
  --data configs\data.yaml `
  --model "$best" `
  --imgsz 640 --conf 0.001 --iou 0.50 --max_det 300 --workers 8
```

▶️ **Produces** (new folder like `runs\eval\YYYYMMDD-HHMMSS_baseline_test\test\`)

* `labels\*.txt` — Ultralytics TXT predictions (normalized to letterboxed frame)
* `predictions.json` / `predictions.ultra.json` — whatever the runner writes (we’ll replace this in step 4)
* `results.csv` (12-number COCO summary; created by the script if missing)
* `..\manifest.json` (one level up) — tiny manifest pointing to these files

> The Ultralytics console header you see (P/R/mAP…) is correct for their letterboxed eval. For **COCO with original pixels**, do steps 3–5.

---

# 3) Make pixel-space detections (our standalone predictor)

This is the **accurate COCO** detections file (xywh in original pixels).

```powershell
$OUT = "runs\predict\baseline_test"
$GT  = "data\raw\annotations\val\nightowls_validation.json"  # NightOwls “test” == validation JSON

python .\scripts\yolo_runs\predict_to_coco.py `
  --data configs\data.yaml `
  --gt $GT `
  --model "$best" `
  --split test `
  --imgsz 640 --conf 0.001 --iou 0.50 --max_det 300 `
  --outdir $OUT
```

▶️ **Produces**

* `runs\predict\baseline_test\predictions.json` — **use this** for scoring

---

# 4) Drop those predictions into the eval folder

Replace the predictions.json produced in Step 2, and replace it with the ones being generated here (can do this manually).

---

# 5) Generate COCO tables + PR tensor

This reads the `predictions.json` that’s inside the eval folder and writes the artifacts.

```powershell
python .\scripts\yolo_runs\run_yolo_baseline.py report --split test
```

▶️ **Produces** (in that same eval **\test** folder)

* `metrics_by_class_area.csv` — per-class AP by size (S/M/L)
* `pr_data.npz` — NumPy archive with precision–recall arrays (plot PR curves from this)

---

# 6) Miss Rate (NightOwls MR-2, “Reasonable”)

Scores **pedestrian only** (cat\_id=1), **h≥50**, **occluded excluded** by default.

```powershell
python .\scripts\metrics\missrate_nightowls.py `
  --gt data\raw\annotations\val\nightowls_validation.json `
  --pred runs\predict\baseline_test\predictions.json `
  --iou 0.5 --min_height 50
```

▶️ **Produces**

* Console line like:
  `MR-2=21.31% | IoU=0.5 | h>=50 | exclude_occluded=True | GT pos=7649 | images=51848 | TP=6859 FP=55001`
* (optional) save it:
  `... | Tee-Object runs\predict\baseline_test\mr2_summary.txt`

> Variants: add `--include_occluded` for R+O; use `--min_height 0` for “All pedestrians”.

---

# 7) Rinse & repeat for image-enhancement variants

For each variant (e.g., CLAHE / Zero-DCE(++)):

* keep **the same** `--imgsz 640 --conf 0.001 --iou 0.50 --max_det 300` (no TTA),
* redo **steps 3 → 6** (you can reuse the eval folder, just overwrite `predictions.json` before running `report`),
* compare `metrics_by_class_area.csv` + **MR-2**.

---

## Sanity checklist (quick)

* After step 3: `[predict_to_coco] wrote N dets from M images → runs\predict\baseline_test\predictions.json`
* After step 5: `metrics_by_class_area.csv` & `pr_data.npz` exist in your latest eval folder.
* MR-2 looks reasonable (e.g., your 21.31% with h≥50, no occlusion).

## Common pitfalls (and fixes)

* **All zeros AP** → you evaluated the TXT/letterbox coords: make sure step 4 overwrote `predictions.json` in the eval folder with the one from step 3 before running `report`.
* **Too many open files** on Windows while predicting → keep `--workers` small (0–2) in `predict_to_coco.py`.
* **“file not in GT (by basename)”** → ensure `--split` and `--gt` json match the same split.

that’s it — follow those 6–7 steps and you’ll land in the exact same place every time, with all artefacts in predictable folders.
