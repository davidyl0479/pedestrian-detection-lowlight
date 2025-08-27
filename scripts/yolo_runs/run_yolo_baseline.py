#!/usr/bin/env python3
"""
NightOwls YOLO Baseline Runner (fully commented)

Purpose
=======
A small, repeatable runner for training and evaluating a **baseline YOLO** model on
NightOwls, with guardrails so you *always* save the right artefacts for post‑hoc analysis.

Design goals
------------
- **Reproducible:** snapshot environment (env.json, pip_freeze) and CLI args.
- **Safe defaults:** no enhancements, optional rectangular training (aspect‑ratio preserved),
  fixed NMS/conf settings at evaluation, no test‑time augmentation.
- **Batteries included:** preflight label validation; eval *always* writes predictions.json,
  per‑image TXT (with confidences), plots, and a manifest with the exact eval knobs.
- **Post‑hoc friendly:** a `report` subcommand that computes per‑class AP and COCO S/M/L AP
  and saves raw PR tensors for custom plotting later.

Key folders (relative to repo root)
-----------------------------------
- `runs/baseline/<TIMESTAMP>_baseline_yv8s_<img>_e<ep>_s<seed>_rect<0|1>/`  → training outputs
- `runs/eval/<TRAIN_TAG>_<val|test>/`                                        → evaluation outputs
- `runs/reports/...`                                                          → (reserved for future)

Dependencies
------------
- `pip install ultralytics pycocotools` (and `pandas` for the report step)

Typical usage
-------------
  python run_yolo_baseline.py preflight --data configs/data.yaml
  python run_yolo_baseline.py train --data configs/data.yaml --weights yolov8s.pt --imgsz 640 \
                                    --epochs 150 --batch 32 --seed 42 --rect
  python run_yolo_baseline.py val   --data configs/data.yaml --imgsz 640 --conf 0.25 --iou 0.70
  python run_yolo_baseline.py test  --data configs/data.yaml --imgsz 640 --conf 0.25 --iou 0.70
  python run_yolo_baseline.py report --split test
"""

from __future__ import annotations
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
import math
import yaml, io

from collections import Counter

import logging

DEBUG = True  # toggle off when you’re done debugging

LOG = logging.getLogger("nightowls.baseline")
LOG.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
LOG.addHandler(_handler)

# Third‑party dependency: Ultralytics YOLO (v8+)
try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    LOG.error("Ultralytics not installed. Run: pip install ultralytics")
    raise

# ------------------------------------------------------------
# Constants / Project paths
# ------------------------------------------------------------
RUNS_BASE = Path("runs")
BASELINE_DIR = RUNS_BASE / "baseline"  # training runs live here
TRAIN_DIR = RUNS_BASE / "train"  # NEW: future training runs go here
EVAL_DIR = RUNS_BASE / "eval"  # evaluation runs live here
REPORTS_DIR = RUNS_BASE / "reports"  # reserved for future reports

# Label & annotation roots (as per your project layout)
LABEL_ROOT = Path("data/raw/labels")
ANN_TRAIN = Path("data/raw/annotations/train/nightowls_training.json")
ANN_VAL = Path("data/raw/annotations/val/nightowls_validation.json")

# Only these class IDs should exist in labels (YOLO txt):
# 0=pedestrian, 1=bicycledriver, 2=motorbikedriver
ALLOWED_CLASSES = {0, 1, 2}

# Tolerances for preflight rounding fixes
# Values that exceed [0,1] by <= EPS_WARN are **warned and auto-clamped**.
# Anything beyond that is treated as an error.
EPS_WARN = 1e-4  # generous tolerance for tiny rounding drift (~0.0001)
EPS_MIN_WH = 1e-6  # ensure strictly positive width/height after clamping


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------


def now_id() -> str:
    """Timestamp string to create unique run folders (YYYYmmdd-HHMMSS)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_root(root: Path, tag: str) -> Path:
    """Create a run directory with standard subfolders.

    Each run gets `logs/` and `artifacts/` placeholders:
    - `logs/`     → reserved for future file logging (we currently log to console).
    - `artifacts/`→ reserved for extra exports (ONNX, parquet detections, etc.).
    """
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{now_id()}_{tag}"
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_env(run_dir: Path, seed: int | None) -> None:
    """Save environment metadata for reproducibility.

    Writes:
    - `env.json`      → Python/Torch/Ultralytics versions, CUDA/GPU info, seed, platform.
    - `pip_freeze.txt`→ exact pip package versions.
    """
    env = {
        "seed": seed,
        "python": sys.version,
        "platform": sys.platform,
    }
    try:
        import torch, platform, ultralytics

        env.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cudnn": torch.backends.cudnn.version()
                if torch.cuda.is_available()
                else None,
                "ultralytics": ultralytics.__version__,
                "machine": platform.machine(),
                "processor": platform.processor(),
            }
        )
        if torch.cuda.is_available():
            env["cuda_device_count"] = torch.cuda.device_count()
            env["gpus"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception as e:
        LOG.warning(f"Env snapshot partial: {e}")

    (run_dir / "env.json").write_text(json.dumps(env, indent=2))

    # Capture a full pinned package list for later recreation of the run
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        (run_dir / "pip_freeze.txt").write_text(freeze)
    except Exception as e:
        LOG.warning(f"pip freeze failed: {e}")


def copy_configs(run_dir: Path, cfg_dir: Path = Path("configs")) -> None:
    """Copy all YAMLs from `configs/` into the run, so the run is self‑contained."""
    dst = run_dir / "configs"
    dst.mkdir(parents=True, exist_ok=True)
    for p in cfg_dir.glob("*.yaml"):
        shutil.copy2(p, dst / p.name)


def newest_best_weight() -> Path | None:
    """Find the most recent best.pt under runs/train/** (or legacy runs/baseline/**)."""
    candidates = sorted(TRAIN_DIR.glob("**/train/weights/best.pt"))
    if not candidates:
        candidates = sorted(BASELINE_DIR.glob("**/train/weights/best.pt"))  # legacy
    return candidates[-1] if candidates else None


# ------------------------------------------------------------
# Preflight checks (fast fail on bad labels)
# ------------------------------------------------------------


def preflight_labels(splits: Iterable[str] = ("train", "val")) -> int:
    """Validate and lightly **repair** YOLO txt labels across the requested splits.

    What it does now:
    - Checks per line: 5 fields, class in {0,1,2}, normalized x/y in [0,1], w/h in (0,1].
    - Checks derived corners within [0,1].
    - **Warns + clamps** tiny out-of-range values (<= EPS_WARN) due to rounding drift,
      and rewrites the label file with corrected values (6-decimal formatting).
    - Empty files are *valid* (background negatives).

    Returns number of *errors* found (0 means OK). Warnings do not count as errors.
    """
    errors = 0
    total_warns = 0
    for split in splits:
        for p in (LABEL_ROOT / split).rglob("*.txt"):
            try:
                raw = p.read_text(encoding="utf-8")
            except Exception as e:
                LOG.error(f"[READ] {p}: {e}")
                errors += 1
                continue
            txt = raw.strip()
            if not txt:
                # Empty file → background image with no GT; expected and fine
                continue

            lines = txt.splitlines()
            out_lines = []
            file_modified = False

            for ln, line in enumerate(lines, start=1):
                parts = line.strip().split()
                if len(parts) != 5:
                    LOG.error(
                        f"[FIELDS] {p}:{ln} expected 5 fields, got {len(parts)} -> {line!r}"
                    )
                    errors += 1
                    continue
                try:
                    cls = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:])
                except Exception as e:
                    LOG.error(f"[PARSE] {p}:{ln} cannot parse -> {line!r} ({e})")
                    errors += 1
                    continue

                if cls not in ALLOWED_CLASSES:
                    LOG.error(
                        f"[CLASS] {p}:{ln} invalid class {cls}; allowed {sorted(ALLOWED_CLASSES)}"
                    )
                    errors += 1

                # Helper to clamp with warning if within tolerance
                def clamp01_with_warn(val: float, what: str) -> tuple[float, bool]:
                    clamped = min(1.0, max(0.0, val))
                    delta = val - clamped
                    if delta == 0.0:
                        return val, False
                    if abs(delta) <= EPS_WARN:
                        LOG.warning(
                            f"[CLAMP] {p}:{ln} {what} {val:.9f} -> {clamped:.6f} (rounding)"
                        )
                        return clamped, True
                    else:
                        LOG.error(f"[RANGE] {p}:{ln} {what} out of [0,1]: {val:.9f}")
                        return val, None  # signal hard error

                # Clamp x,y,w,h if tiny drift; error if far out
                status_error = False
                changed = False
                for key in ("x", "y", "w", "h"):
                    pass
                x_new, ch = clamp01_with_warn(x, "x")
                changed |= ch or False
                status_error |= ch is None
                y_new, ch = clamp01_with_warn(y, "y")
                changed |= ch or False
                status_error |= ch is None
                w_new, ch = clamp01_with_warn(w, "w")
                changed |= ch or False
                status_error |= ch is None
                h_new, ch = clamp01_with_warn(h, "h")
                changed |= ch or False
                status_error |= ch is None
                x, y, w, h = x_new, y_new, w_new, h_new

                # Enforce strictly positive w/h
                if w <= 0 or h <= 0:
                    # Try to lift tiny non-positive values due to rounding
                    if -EPS_WARN <= w <= 0:
                        LOG.warning(
                            f"[CLAMP] {p}:{ln} w {w:.9f} -> {EPS_MIN_WH:.6f} (min width)"
                        )
                        w = EPS_MIN_WH
                        changed = True
                    else:
                        LOG.error(f"[RANGE] {p}:{ln} non-positive width w={w:.9f}")
                        status_error = True
                    if -EPS_WARN <= h <= 0:
                        LOG.warning(
                            f"[CLAMP] {p}:{ln} h {h:.9f} -> {EPS_MIN_WH:.6f} (min height)"
                        )
                        h = EPS_MIN_WH
                        changed = True
                    else:
                        if h <= 0 and not (-EPS_WARN <= h <= 0):
                            LOG.error(f"[RANGE] {p}:{ln} non-positive height h={h:.9f}")
                            status_error = True

                # Check/repair corners if barely out
                x1, y1 = x - w / 2, y - h / 2
                x2, y2 = x + w / 2, y + h / 2

                def corner_fix(a1, a2, name) -> tuple[float, float, bool, bool]:
                    """Clamp corners to [0,1] if the overflow is tiny.
                    Returns (c1,c2, changed, hard_error)
                    """
                    c1 = min(1.0, max(0.0, a1))
                    c2 = min(1.0, max(0.0, a2))
                    over = max(abs(a1 - c1), abs(a2 - c2))
                    if over == 0.0:
                        return a1, a2, False, False
                    if over <= EPS_WARN:
                        LOG.warning(
                            f"[CLAMP] {p}:{ln} {name} corners -> within [0,1] (rounding)"
                        )
                        return c1, c2, True, False
                    else:
                        LOG.error(
                            f"[CORNERS] {p}:{ln} {name} corners out of bounds by {over:.6f}"
                        )
                        return a1, a2, False, True

                x1c, x2c, chx, errx = corner_fix(x1, x2, "x")
                y1c, y2c, chy, erry = corner_fix(y1, y2, "y")
                changed |= chx or chy
                status_error |= errx or erry

                if chx or chy:
                    # Recompute (x,y,w,h) from clamped corners; ensure min positive w/h
                    w = max(x2c - x1c, EPS_MIN_WH)
                    h = max(y2c - y1c, EPS_MIN_WH)
                    x = (x1c + x2c) / 2
                    y = (y1c + y2c) / 2

                if status_error:
                    errors += 1
                    # keep original line so you can inspect later
                    out_lines.append(line)
                    continue

                if changed:
                    total_warns += 1
                    file_modified = True
                out_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            # If we made safe corrections, rewrite the file preserving a trailing newline
            if file_modified:
                new_text = "\n".join(out_lines) + "\n"
                try:
                    p.write_text(new_text, encoding="utf-8")
                    LOG.warning(
                        f"[REWRITE] {p} (applied {sum(1 for _ in out_lines)} lines with safe clamps)"
                    )
                except Exception as e:
                    LOG.error(f"[WRITE] {p}: failed to rewrite after clamps ({e})")
                    errors += 1

    if errors == 0:
        if total_warns:
            LOG.info(
                f"Preflight labels: OK with {total_warns} warn+clamp adjustments (<= {EPS_WARN})"
            )
        else:
            LOG.info("Preflight labels: OK")
    else:
        LOG.error(f"Preflight labels: {errors} problems found")
    return errors


# ------------------------------------------------------------
# Train / Eval commands
# ------------------------------------------------------------


def train(args) -> None:
    """Train a baseline YOLO model.

    - Preserves aspect ratio if `--rect` is passed (and explicitly disables mosaic).
    - Saves args/env/configs snapshots into the run directory for reproducibility.
    - Leaves Ultralytics defaults for losses/augs (baseline purity: no enhancement tricks).
    """
    # Auto-tag: "<weights-stem>_<domain>_<img>_e<ep>_s<seed>_rect<0|1>"
    from pathlib import Path as _P

    model_name = _P(args.weights).stem  # e.g., yolo11s (or yolov8s, etc.)
    domain = "zerodce" if "zerodce" in str(args.data).lower() else "raw"
    tag = f"{model_name}_{domain}_{args.imgsz}_e{args.epochs}_s{args.seed}_rect{int(args.rect)}"
    run_root = make_run_root(TRAIN_DIR, tag)

    # Persist config + env + args (so each run stands alone)
    (run_root / "args.json").write_text(json.dumps(vars(args), indent=2))
    snapshot_env(run_root, args.seed)
    copy_configs(run_root)

    # Validate labels before spending GPU time
    if not args.skip_preflight:
        errs = preflight_labels(("train", "val"))
        if errs:
            LOG.error("Aborting due to preflight errors.")
            sys.exit(2)

    # Best‑effort seeding (avoid forcing slow deterministic kernels)
    try:
        import torch, numpy as np

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        # torch.use_deterministic_algorithms(False)
    except Exception as e:
        LOG.warning(f"Seed setup warning: {e}")

    LOG.info("Loading model…")
    model = YOLO(args.weights)

    LOG.info("Starting training… (rect=%s, mosaic=0.0)", args.rect)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        workers=args.workers,
        project=str(run_root),
        name="train",  # Ultralytics will create run_root/train/
        exist_ok=True,
        verbose=True,
        rect=args.rect,  # preserve aspect ratio within batches
        mosaic=0.0,  # disable mosaic explicitly when rect=True
    )

    # Where Ultralytics stored things (weights, results.csv, plots…)
    best = Path(results.save_dir) / "weights" / "best.pt"
    last = Path(results.save_dir) / "weights" / "last.pt"

    # Minimal manifest so later steps can quickly locate weights
    manifest = {
        "train_dir": str(results.save_dir),
        "best": str(best),
        "last": str(last),
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    LOG.info(f"Training finished. Weights: {best}")


def _eval_common(args, split: str) -> None:
    """Shared evaluator for `val` and `test` with official-ID predictions.json and auto results.csv."""
    assert split in ("val", "test")

    model_path = Path(args.model) if args.model else newest_best_weight()
    if not model_path or not model_path.exists():
        LOG.error("No trained model found. Provide --model path or train first.")
        sys.exit(2)

    tag = f"{model_path.parent.parent.parent.name}_{split}"
    run_root = make_run_root(EVAL_DIR, tag)

    (run_root / "args.json").write_text(json.dumps(vars(args), indent=2))
    snapshot_env(run_root, args.seed)
    copy_configs(run_root)

    LOG.info(f"Evaluating split={split} with fixed NMS/conf (no TTA)…")
    model = YOLO(str(model_path))
    out = model.val(
        data=args.data,
        split=split,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
        save_txt=True,
        save_conf=True,
        save_json=True,
        plots=True,
        project=str(run_root),
        name=split,
        exist_ok=True,
    )

    eval_dir = Path(out.save_dir)
    labels_dir = eval_dir / "labels"

    # --- Build official predictions.json (original TXT-based path) ---

    # Decide which GT JSON to use for this split.
    # For 'test' we keep VAL GT. For 'val' we detect if the val entry points to TRAIN.
    from pathlib import Path as _P

    def _val_points_to_train(data_yaml_path: str) -> bool:
        try:
            dy = yaml.safe_load(Path(data_yaml_path).read_text(encoding="utf-8"))
            val_entry = str(dy.get("val", ""))
            val_entry_norm = val_entry.replace("\\", "/")
            # If it's an index file, peek some lines
            if val_entry_norm.endswith(".txt"):
                with open(val_entry_norm, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if "/images/train/" in line.replace("\\", "/"):
                            return True
                        if i > 2000:  # don’t scan huge files fully
                            break
            # Or if the path itself contains /images/train/
            return "/images/train/" in val_entry_norm
        except Exception:
            return False

    use_train_gt = (split == "val") and _val_points_to_train(args.data)

    gt_json = ANN_TRAIN if use_train_gt else ANN_VAL
    gt = json.loads(gt_json.read_text(encoding="utf-8"))

    # filename -> (image_id, width, height)
    im_map = {
        _P(im["file_name"]).name: (im["id"], im.get("width"), im.get("height"))
        for im in gt["images"]
    }

    # original fixed mapping (YOLO idx -> NightOwls category_id)
    cat_map = {0: 1, 1: 2, 2: 3}

    official = []
    for p in sorted(labels_dir.glob("*.txt")):
        base = p.stem
        fname = base + ".png"
        if fname not in im_map:
            fname = base + ".jpg"
            if fname not in im_map:
                continue
        img_id, W, H = im_map[fname]

        for line in p.read_text().splitlines():
            # cls, conf, xc, yc, w, h  (your original ordering)
            cls, conf, xc, yc, bw, bh = map(float, line.split())
            cid = cat_map.get(int(cls))
            if cid is None:
                continue
            # normalized xywh -> pixel xywh
            x = (xc - bw / 2.0) * W
            y = (yc - bh / 2.0) * H
            w = bw * W
            h = bh * H
            official.append(
                {
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": [x, y, w, h],
                    "score": conf,
                }
            )

    # Keep Ultralytics' original, but make official one canonical
    try:
        (eval_dir / "predictions.json").rename(eval_dir / "predictions.ultra.json")
    except Exception:
        pass
    pred_json = eval_dir / "predictions.json"
    pred_json.write_text(json.dumps(official))

    # --- COCOeval on the official predictions, restricted to the subset we scored ---
    cocoGt = COCO(str(gt_json))
    cocoGt.dataset.setdefault("info", {})
    for ann in cocoGt.dataset.get("annotations", []):
        ann.setdefault("iscrowd", 0)
    cocoGt.createIndex()

    cocoDt = cocoGt.loadRes(str(pred_json))
    ev = COCOeval(cocoGt, cocoDt, "bbox")
    # Restrict evaluation to ONLY the images we predicted on (internal-val subset)
    ev.params.imgIds = sorted({int(d["image_id"]) for d in official})
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = (
        ev.stats
    )  # [AP50-95, AP50, AP75, AP_S, AP_M, AP_L, AR_1, AR_10, AR_100, AR_S, AR_M, AR_L]

    # Ensure results.csv exists (COCOeval)
    # Write COCO summary to results.csv (kept in sync with predictions.json)
    res_csv = eval_dir / "results.csv"
    try:
        import csv

        with res_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "AP50-95",
                    "AP50",
                    "AP75",
                    "AP_S",
                    "AP_M",
                    "AP_L",
                    "AR_1",
                    "AR_10",
                    "AR_100",
                    "AR_S",
                    "AR_M",
                    "AR_L",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "AP50-95": s[0],
                    "AP50": s[1],
                    "AP75": s[2],
                    "AP_S": s[3],
                    "AP_M": s[4],
                    "AP_L": s[5],
                    "AR_1": s[6],
                    "AR_10": s[7],
                    "AR_100": s[8],
                    "AR_S": s[9],
                    "AR_M": s[10],
                    "AR_L": s[11],
                }
            )
    except Exception as e:
        LOG.warning(f"Could not write results.csv from COCOeval: {e}")
        res_csv.write_text("note,results.csv write failed\n")

    # Manifest (points to official predictions.json)
    manifest = {
        "eval_dir": str(eval_dir),
        "model": str(model_path),
        "split": split,
        "predictions_json": str(pred_json),
        "results_csv": str(res_csv),
        "nms": {
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "agnostic_nms": bool(args.agnostic_nms),
        },
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    LOG.info(f"{split} evaluation complete. Outputs in {eval_dir}")


def eval_val(args) -> None:
    """Evaluate on your **internal validation** split (split=val)."""
    _eval_common(args, split="val")


def eval_test(args) -> None:
    """Evaluate on the **final test** set (NightOwls val images; split=test)."""
    _eval_common(args, split="test")


# ------------------------------------------------------------
# Reporting (post‑hoc) using pycocotools
# ------------------------------------------------------------


def report(args) -> None:
    """Post‑hoc reporting after an eval run.

    Produces:
    - `metrics_by_class_area.csv` → per‑class AP50‑95 and AP by COCO size bins (S/M/L).
    - `pr_data.npz`               → raw precision tensor + IoU/Recall grids for custom plots.

    Notes:
    - Uses COCOeval, reading GT json (NightOwls) and the saved predictions.json.
    - For slicing by pose_id or image characteristics later, add a new subcommand that
      filters annotations or image IDs and re‑runs COCOeval (pattern shown earlier).
    """

    # Locate eval dir: explicit --run overrides newest-by-split
    split = args.split
    eval_dir = Path(args.run) if getattr(args, "run", None) else None
    if not eval_dir:
        eval_dirs = sorted(EVAL_DIR.glob(f"*_{split}"))
        if not eval_dirs:
            LOG.error(
                f"No eval runs found for split={split}. Run val/test first or pass --run."
            )
            sys.exit(2)
        eval_dir = eval_dirs[-1]

    # Load the eval manifest to locate predictions.json
    man_path = eval_dir / "manifest.json"
    if not man_path.exists():
        LOG.error("manifest.json not found in eval dir.")
        sys.exit(2)
    manifest = json.loads(man_path.read_text())
    pred_json = Path(manifest["predictions_json"])

    # Choose the appropriate GT json for the split
    with open(args.data, "r", encoding="utf-8") as f:
        dy = yaml.safe_load(f)
    val_entry = str(dy.get("val", ""))

    def _points_to_train(p: str) -> bool:
        # If it's an index file, peek its contents; otherwise just string-check the path
        try:
            if p.endswith(".txt"):
                with open(p, "r", encoding="utf-8") as ff:
                    for i, line in enumerate(ff):
                        if "/images/train/" in line.replace("\\", "/"):
                            return True
                        if i > 2000:  # don't scan huge files fully
                            break
        except Exception:
            pass
        return "/images/train/" in p.replace("\\", "/")

    use_train_gt = (split == "val") and _points_to_train(val_entry)
    gt_json = ANN_TRAIN if use_train_gt else ANN_VAL

    if not gt_json.exists():
        LOG.error(f"GT not found: {gt_json}")
        sys.exit(2)

    # Standard COCO evaluation (NightOwls GT needs small patches)
    cocoGt = COCO(str(gt_json))
    cocoGt.dataset.setdefault("info", {})
    for ann in cocoGt.dataset.get("annotations", []):
        ann.setdefault("iscrowd", 0)
    cocoGt.createIndex()

    cocoDt = cocoGt.loadRes(str(pred_json))
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    # Limit evaluation to the images we actually predicted on
    pred_img_ids = sorted({d["image_id"] for d in json.loads(pred_json.read_text())})
    cocoEval.params.imgIds = pred_img_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # Write/refresh results.csv from the current COCO summary so it never stays stale
    import csv

    s = (
        cocoEval.stats
    )  # [AP50-95, AP50, AP75, AP_S, AP_M, AP_L, AR_1, AR_10, AR_100, AR_S, AR_M, AR_L]
    out_csv_path = eval_dir / "results.csv"
    with out_csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "AP50-95",
                "AP50",
                "AP75",
                "AP_S",
                "AP_M",
                "AP_L",
                "AR_1",
                "AR_10",
                "AR_100",
                "AR_S",
                "AR_M",
                "AR_L",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "AP50-95": s[0],
                "AP50": s[1],
                "AP75": s[2],
                "AP_S": s[3],
                "AP_M": s[4],
                "AP_L": s[5],
                "AR_1": s[6],
                "AR_10": s[7],
                "AR_100": s[8],
                "AR_S": s[9],
                "AR_M": s[10],
                "AR_L": s[11],
            }
        )

    # Extract per‑class AP (all) and by COCO size bins (S/M/L)
    pr = cocoEval.eval["precision"]  # shape: [IoU, Recall, Class, Area, MaxDet]
    ious = cocoEval.params.iouThrs
    recs = cocoEval.params.recThrs
    names = [cocoGt.cats[k]["name"] for k in sorted(cocoGt.cats.keys())]

    import numpy as np

    ap_all = pr[:, :, :, 0, -1].mean(axis=(0, 1))  # area=all
    ap_s = pr[:, :, :, 1, -1].mean(axis=(0, 1))  # area=small
    ap_m = pr[:, :, :, 2, -1].mean(axis=(0, 1))  # area=medium
    ap_l = pr[:, :, :, 3, -1].mean(axis=(0, 1))  # area=large

    rows = []
    for k, name in enumerate(names):
        rows.append(
            {
                "class": name,
                "AP50-95": float(ap_all[k]),
                "AP_S": float(ap_s[k]),
                "AP_M": float(ap_m[k]),
                "AP_L": float(ap_l[k]),
            }
        )

    # Save a tidy CSV for quick inspection in notebooks/spreadsheets
    import pandas as pd

    df = pd.DataFrame(rows)
    out_csv = eval_dir / "metrics_by_class_area.csv"
    df.to_csv(out_csv, index=False)

    # Save raw PR arrays (so you can draw custom PR/iso‑F1 plots without re‑running eval)
    npz_path = eval_dir / "pr_data.npz"
    import numpy as np

    np.savez(npz_path, precision=pr, ious=ious, recalls=recs)

    LOG.info(f"Saved: {out_csv}")
    LOG.info(f"Saved raw PR arrays: {npz_path}")


# ------------------------------------------------------------
# CLI wiring
# ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Define CLI with clear subcommands:
    - preflight: quick label validation
    - train:     baseline training
    - val/test:  evaluation with fixed NMS and required artefacts
    - report:    post‑hoc metrics export (per‑class + S/M/L, PR tensor)
    """
    p = argparse.ArgumentParser(description="NightOwls YOLO Baseline Runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- preflight ---
    sp = sub.add_parser("preflight", help="Validate label format & ranges")
    sp.add_argument(
        "--data", default="configs/data.yaml", help="Not used (kept for symmetry)"
    )

    # --- train ---
    st = sub.add_parser("train", help="Train baseline model")
    st.add_argument(
        "--data", required=True, help="Path to Ultralytics data.yaml (lists & names)"
    )
    st.add_argument(
        "--weights", default="yolov8s.pt", help="Pretrained weights to fine‑tune"
    )
    st.add_argument("--epochs", type=int, default=150)
    st.add_argument("--imgsz", type=int, default=640)
    st.add_argument("--batch", type=int, default=32)
    st.add_argument("--workers", type=int, default=8)
    st.add_argument("--device", default=0)
    st.add_argument("--seed", type=int, default=42)
    st.add_argument(
        "--rect", action="store_true", help="Use rectangular training (preserve AR)"
    )
    st.add_argument(
        "--skip_preflight",
        action="store_true",
        help="Skip label checks (not recommended)",
    )

    # --- val ---
    sv = sub.add_parser("val", help="Evaluate on internal validation (split=val)")
    sv.add_argument("--data", required=True)
    sv.add_argument("--model", help="Path to .pt; defaults to newest baseline best.pt")
    sv.add_argument("--imgsz", type=int, default=640)
    sv.add_argument("--device", default=0)
    sv.add_argument("--workers", type=int, default=8)
    sv.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="NMS confidence threshold (fixed across runs)",
    )
    sv.add_argument(
        "--iou", type=float, default=0.70, help="NMS IoU threshold (fixed across runs)"
    )
    sv.add_argument("--max_det", type=int, default=300)
    sv.add_argument("--agnostic_nms", action="store_true")
    sv.add_argument("--seed", type=int, default=42)

    # --- test --- (clone val args; evaluates NightOwls val images as the final test set)
    stt = sub.add_parser(
        "test", help="Evaluate on final test (NightOwls val; split=test)"
    )
    for a in sv._actions[1:]:  # clone everything except the built‑in help action
        if a.dest != "help":
            stt._add_action(a)

    # --- report ---
    rp = sub.add_parser("report", help="Post-hoc reporting (per-class, S/M/L)")
    rp.add_argument("--split", choices=["val", "test"], default="test")
    rp.add_argument(
        "--data",
        default="configs/data.yaml",
        help="Path to data.yaml (used to infer whether val points to TRAIN or VAL)",
    )
    rp.add_argument(
        "--run",
        type=str,
        help="Optional: path to a specific eval run dir (…/runs/eval/<timestamp>_<tag>_<split>)",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "preflight":
        errs = preflight_labels(("train", "val"))
        sys.exit(0 if errs == 0 else 2)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "val":
        eval_val(args)
    elif args.cmd == "test":
        eval_test(args)
    elif args.cmd == "report":
        report(args)


if __name__ == "__main__":
    main()
