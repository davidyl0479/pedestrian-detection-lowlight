#!/usr/bin/env python3
"""
prepare_nightowls.py — Data prep CLI for NightOwls low-light detection experiments.

Stdlib-only. Creates YOLO label files (with all `ignore==1` dropped), builds deterministic
train/val/test index lists, assembles a 3:1 mixed training list, optionally selects tuning
subsets (random or metrics-filtered), and emits ready-to-run Ultralytics YAML configs.
It also provides a simple audit to verify label presence for images referenced by the
index lists.

Updates:
- Now supports **custom annotation filenames/locations** via `--train-ann-file` and
  `--val-ann-file`. If not provided, the script auto-resolves by looking for
  `train.json`/`val.json` under `--raw-ann`, *or* a single `*.json` inside
  `--raw-ann/train/` and `--raw-ann/val/`.

"""

from __future__ import annotations

# --- Standard library imports only to keep the script portable ---
import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

# ===============================================================
# Paths & constants
# ===============================================================

# Treat the current working directory (project root) as the anchor for relative paths.
PROJECT_ROOT = Path(".")

# Default locations; can be overridden via CLI flags.
DEF_RAW_ANN = PROJECT_ROOT / "data/raw/annotations"
DEF_RAW_IMG = PROJECT_ROOT / "data/raw/images"
DEF_OUT_LABEL = PROJECT_ROOT / "data/processed/labels"  # canonical generated labels
DEF_MIRROR_LABEL = PROJECT_ROOT / "data/raw/labels"  # mirrored labels for Ultralytics
DEF_INDEX_DIR = PROJECT_ROOT / "index"
DEF_CONFIGS_DIR = PROJECT_ROOT / "configs"
DEF_LOG_DIR = PROJECT_ROOT / "logs"

# Category mapping (NightOwls): raw COCO category ids -> YOLO class ids
# We include only the 3 usable classes and map them to {0,1,2}
RAW_TO_YOLO = {1: 0, 2: 1, 3: 2}  # pedestrian, bicycledriver, motorbikedriver
YOLO_NAMES = {0: "pedestrian", 1: "bicycledriver", 2: "motorbikedriver"}


# ===============================================================
# Small utilities
# ===============================================================


def ensure_dir(p: Path) -> None:
    """Create a directory (and parents) if it doesn't already exist."""
    p.mkdir(parents=True, exist_ok=True)


def fmt6(x: float) -> str:
    """Format a float with 6 decimal places for YOLO label files."""
    return f"{x:.6f}"


def md5_hex(s: str) -> str:
    """Deterministic MD5 hex digest of a string; used for stable ordering/splits."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def md5_mod(s: str, mod: int) -> int:
    """Hash a string and return an integer bucket in [0, mod)."""
    return int(md5_hex(s), 16) % mod


def relpath(path: Path) -> str:
    """Return a POSIX-style path relative to PROJECT_ROOT (portable across OSes)."""
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def resolve_ann_path(raw_ann_root: Path, split: str, explicit: Optional[str]) -> Path:
    """Resolve the annotation JSON path for a split.

    Resolution order:
      1) If `explicit` is provided, use it directly.
      2) If `<raw_ann_root>/<split>.json` exists, use it.
      3) If `<raw_ann_root>/<split>/` exists and contains **exactly one** `*.json`, use it.

    Raises FileNotFoundError/ValueError with a clear message otherwise.
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(
                f"Provided {split} annotations not found: {explicit}"
            )
        return p

    # Try flat file: data/raw/annotations/train.json or val.json
    flat = raw_ann_root / f"{split}.json"
    if flat.exists():
        return flat

    # Try nested folder with a single JSON: data/raw/annotations/train/*.json
    nested_dir = raw_ann_root / split
    if nested_dir.exists() and nested_dir.is_dir():
        jsons = sorted(nested_dir.glob("*.json"))
        if len(jsons) == 1:
            return jsons[0]
        elif len(jsons) == 0:
            raise FileNotFoundError(
                f"No JSON found under {relpath(nested_dir)} for split '{split}'."
            )
        else:
            raise ValueError(
                f"Multiple JSON files under {relpath(nested_dir)}; please pass --{split}-ann-file explicitly."
            )

    raise FileNotFoundError(
        f"Could not resolve {split} annotations. Provide --{split}-ann-file or place a JSON at "
        f"{relpath(flat)} or exactly one JSON inside {relpath(nested_dir)}"
    )


# ===============================================================
# COCO JSON I/O
# ===============================================================


def load_coco(
    json_path: Path,
) -> Tuple[Dict[int, dict], Dict[int, List[dict]], Dict[int, str]]:
    """Load a COCO-style JSON file and return image/annotation/category mappings."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images_by_id = {img["id"]: img for img in data.get("images", [])}
    anns_by_image: Dict[int, List[dict]] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    cat_names = {
        c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])
    }
    return images_by_id, anns_by_image, cat_names


def clamp_bbox_xywh(
    x: float, y: float, w: float, h: float, W: int, H: int
) -> Optional[Tuple[float, float, float, float]]:
    """Clamp a COCO-format bbox [x, y, w, h] to image extent. Return None if area <= 0."""
    x2 = x + w
    y2 = y + h
    x = max(0.0, min(float(x), float(W)))
    y = max(0.0, min(float(y), float(H)))
    x2 = max(0.0, min(float(x2), float(W)))
    y2 = max(0.0, min(float(y2), float(H)))
    w = x2 - x
    h = y2 - y
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def coco_to_yolo_lines(anns: List[dict], W: int, H: int) -> Tuple[List[str], int, int]:
    """Convert usable COCO annotations to YOLO label lines for a single image.

    Usable = `ignore == 0` and `category_id in RAW_TO_YOLO`.
    """
    lines: List[str] = []
    kept = 0
    dropped = 0
    for a in anns:
        if int(a.get("ignore", 0)) == 1:
            dropped += 1
            continue
        cid = int(a.get("category_id"))
        if cid not in RAW_TO_YOLO:
            dropped += 1
            continue
        x, y, w, h = a["bbox"]
        clamped = clamp_bbox_xywh(x, y, w, h, W, H)
        if clamped is None:
            dropped += 1
            continue
        x, y, w, h = clamped
        cx = (x + w / 2.0) / float(W)
        cy = (y + h / 2.0) / float(H)
        nw = w / float(W)
        nh = h / float(H)
        line = f"{RAW_TO_YOLO[cid]} {fmt6(cx)} {fmt6(cy)} {fmt6(nw)} {fmt6(nh)}"
        lines.append(line)
        kept += 1
    return lines, kept, dropped


# ===============================================================
# Subcommand: labels — generate YOLO labels and mirror for Ultralytics
# ===============================================================


def cmd_labels(args: argparse.Namespace) -> None:
    raw_ann = Path(args.raw_ann)
    raw_img = Path(args.raw_img)
    out_label = Path(args.out_label)
    mirror_label = Path(args.mirror_label)

    # Resolve annotation JSONs (now flexible)
    train_json = resolve_ann_path(raw_ann, "train", args.train_ann_file)
    val_json = resolve_ann_path(raw_ann, "val", args.val_ann_file)

    # Ensure output directories exist (canonical + mirrored)
    ensure_dir(out_label / "train")
    ensure_dir(out_label / "test")
    ensure_dir(mirror_label / "train")
    ensure_dir(mirror_label / "val")
    ensure_dir(DEF_LOG_DIR)

    missing_log = DEF_LOG_DIR / "missing_images.txt"
    missing_log.write_text("", encoding="utf-8")  # truncate the log each run

    # Load COCO JSONs (raw/val is your final Test set)
    images_train, anns_train, _ = load_coco(train_json)
    images_val, anns_val, _ = load_coco(val_json)

    stats = {
        "train": {
            "images": 0,
            "positives": 0,
            "negatives": 0,
            "boxes_kept": 0,
            "boxes_dropped": 0,
            "missing_imgs": 0,
        },
        "test": {
            "images": 0,
            "positives": 0,
            "negatives": 0,
            "boxes_kept": 0,
            "boxes_dropped": 0,
            "missing_imgs": 0,
        },
    }

    def process_split(
        split_name: str,
        images_by_id: Dict[int, dict],
        anns_by_image: Dict[int, List[dict]],
        proc_dir: Path,
        mirror_dir: Path,
        img_subdir: str,
    ) -> None:
        for img_id, img in images_by_id.items():
            file_name = img["file_name"]
            W = int(img.get("width"))
            H = int(img.get("height"))
            img_path = raw_img / img_subdir / file_name

            if not img_path.exists():
                with open(missing_log, "a", encoding="utf-8") as mf:
                    mf.write(relpath(img_path) + "\n")
                stats[split_name]["missing_imgs"] += 1

            lines, kept, dropped = coco_to_yolo_lines(
                anns_by_image.get(img_id, []), W, H
            )

            stats[split_name]["images"] += 1
            stats[split_name]["positives"] += int(bool(lines))
            stats[split_name]["negatives"] += int(not bool(lines))
            stats[split_name]["boxes_kept"] += kept
            stats[split_name]["boxes_dropped"] += dropped

            stem = Path(file_name).with_suffix("").name
            out_proc = proc_dir / f"{stem}.txt"
            out_mirror = mirror_dir / f"{stem}.txt"
            ensure_dir(out_proc.parent)
            ensure_dir(out_mirror.parent)
            content = "\n".join(lines)
            out_proc.write_text(content, encoding="utf-8")
            out_mirror.write_text(content, encoding="utf-8")

    process_split(
        "train",
        images_train,
        anns_train,
        out_label / "train",
        mirror_label / "train",
        "train",
    )
    process_split(
        "test", images_val, anns_val, out_label / "test", mirror_label / "val", "val"
    )

    print("\n[labels] Summary:")
    for split in ("train", "test"):
        s = stats[split]
        print(
            f"  {split}: images={s['images']} (pos={s['positives']}, neg={s['negatives']}), "
            f"boxes_kept={s['boxes_kept']}, boxes_dropped={s['boxes_dropped']}, missing_imgs={s['missing_imgs']}"
        )
    print(
        f"  Labels written to: {relpath(out_label)} (canonical) and mirrored to {relpath(mirror_label)}"
    )


# ===============================================================
# Subcommand: indices — build pools and deterministic splits
# ===============================================================


def cmd_indices(args: argparse.Namespace) -> None:
    raw_ann = Path(args.raw_ann)
    raw_img = Path(args.raw_img)
    index_dir = Path(args.index_dir)
    ensure_dir(index_dir)

    # Resolve the same JSONs used in `labels`
    train_json = resolve_ann_path(raw_ann, "train", args.train_ann_file)
    val_json = resolve_ann_path(raw_ann, "val", args.val_ann_file)

    images_train, anns_train, _ = load_coco(train_json)
    images_val, anns_val, _ = load_coco(val_json)

    def build_flags(
        images_by_id: Dict[int, dict], anns_by_image: Dict[int, List[dict]], split: str
    ) -> List[dict]:
        rows: List[dict] = []
        for img_id, img in images_by_id.items():
            file_name = img["file_name"]
            W = int(img.get("width"))
            H = int(img.get("height"))
            anns = anns_by_image.get(img_id, [])
            has_usable = False
            has_ignore = False
            n_gt = 0
            for a in anns:
                if int(a.get("ignore", 0)) == 1:
                    has_ignore = True
                    continue
                if int(a.get("category_id")) in RAW_TO_YOLO:
                    has_usable = True
                    n_gt += 1
            rows.append(
                {
                    "image_id": img_id,
                    "path": relpath(raw_img / split / file_name),
                    "split": split,
                    "has_usable_gt": has_usable,
                    "has_ignore": has_ignore,
                    "n_gt": n_gt,
                    "width": W,
                    "height": H,
                }
            )
        return rows

    rows_train = build_flags(images_train, anns_train, "train")
    rows_val = build_flags(images_val, anns_val, "val")

    # Catalog CSV
    csv_path = index_dir / "images_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows_train[0].keys())
            if rows_train
            else list(rows_val[0].keys()),
        )
        writer.writeheader()
        if rows_train:
            writer.writerows(rows_train)
        if rows_val:
            writer.writerows(rows_val)

    # Pools from raw/train
    positives = [r["path"] for r in rows_train if r["has_usable_gt"]]
    neg_bg_only = [
        r["path"]
        for r in rows_train
        if (not r["has_usable_gt"]) and (not r["has_ignore"])
    ]
    ignore_only = [
        r["path"] for r in rows_train if (not r["has_usable_gt"]) and r["has_ignore"]
    ]

    # Deterministic 90/10 split per pool using MD5(path) % 10
    def split_hash(
        paths: List[str], mod: int = 10, val_bucket: int = 0
    ) -> Tuple[List[str], List[str]]:
        val_list, train_list = [], []
        for p in paths:
            (val_list if md5_mod(p, mod) == val_bucket else train_list).append(p)
        return val_list, train_list

    val_pos, train_pos = split_hash(positives)
    val_neg, train_neg = split_hash(neg_bg_only)

    # Writers for the index files
    def write_list(path: Path, items: Iterable[str]) -> None:
        ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(it + "\n")

    # ~90% training pools
    write_list(index_dir / "train_positives.txt", train_pos)
    write_list(index_dir / "train_negatives_bg_only.txt", train_neg)

    # ~10% internal validation (mix of pos/neg)
    write_list(index_dir / "val_internal.txt", val_pos + val_neg)

    # Final test set: all images from raw/val
    write_list(index_dir / "test_all.txt", [r["path"] for r in rows_val])

    print("[indices] Summary:")
    print(
        f"  train pools: positives={len(train_pos)}, negatives_bg_only={len(train_neg)}"
    )
    print(
        f"  val_internal: positives={len(val_pos)}, negatives_bg_only={len(val_neg)}, total={len(val_pos) + len(val_neg)}"
    )
    print(f"  ignore-only (excluded from training): {len(ignore_only)}")
    print(f"  test_all (raw/val): {len(rows_val)}")
    print(f"  Wrote: {relpath(csv_path)} and lists under {relpath(index_dir)}")


# ===============================================================
# Subcommand: train-mixed — build interleaved training list (e.g., 3:1)
# ===============================================================


def deterministic_sorted(paths: List[str]) -> List[str]:
    """Return paths sorted by MD5(path), providing a stable pseudo-random order."""
    return sorted(paths, key=lambda p: md5_hex(p))


def interleave_ratio(pos: List[str], neg: List[str], p: int, n: int) -> List[str]:
    """Interleave `p` positives followed by `n` negatives repeatedly.

    If the negative pool is shorter, we cycle through it (to maintain the ratio).
    """
    if p <= 0 or n < 0:
        raise ValueError("Invalid ratio")
    pos_sorted = deterministic_sorted(pos)
    neg_sorted = deterministic_sorted(neg) if n > 0 else []
    mixed: List[str] = []
    ni = 0
    for i in range(0, len(pos_sorted), p):
        block = pos_sorted[i : i + p]
        mixed.extend(block)
        for _ in range(n):
            if not neg_sorted:
                break
            mixed.append(neg_sorted[ni])
            ni = (ni + 1) % len(neg_sorted)
    return mixed


def cmd_train_mixed(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    with open(index_dir / "train_positives.txt", "r", encoding="utf-8") as f:
        pos = [ln.strip() for ln in f if ln.strip()]
    with open(index_dir / "train_negatives_bg_only.txt", "r", encoding="utf-8") as f:
        neg = [ln.strip() for ln in f if ln.strip()]

    try:
        p_str, n_str = args.ratio.split(":")
        p, n = int(p_str), int(n_str)
    except Exception as e:
        raise ValueError("--ratio must be like 3:1") from e

    mixed = interleave_ratio(pos, neg, p, n)

    if args.epoch_mult != 1.0 and args.epoch_mult > 0:
        total = int(len(mixed) * args.epoch_mult)
        if total <= len(mixed):
            mixed = mixed[:total]
        else:
            times = total // len(mixed)
            rem = total % len(mixed)
            mixed = mixed * times + mixed[:rem]

    out_path = index_dir / "train_mixed.txt"
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        for pth in mixed:
            f.write(pth + "\n")

    print("[train-mixed] Summary:")
    print(f"  positives in pool: {len(pos)}; negatives in pool: {len(neg)}")
    print(f"  ratio used: {p}:{n}; epoch_mult={args.epoch_mult}")
    print(f"  train_mixed length: {len(mixed)}")
    print(f"  Wrote: {relpath(out_path)}")


# ===============================================================
# Subcommand: tuning — build deterministic or metrics-filtered subsets
# ===============================================================


def load_metrics_jsons(metric_files: List[Path]) -> Dict[str, dict]:
    """Load one or more per-image metrics JSONs into a single dict keyed by file_name."""
    combined: Dict[str, dict] = {}
    for mf in metric_files:
        with open(mf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for fname, record in data.items():
            combined[fname] = record
    return combined


def filter_by_metrics(
    candidates: List[str],
    metrics: Dict[str, dict],
    bounds: Dict[str, Tuple[Optional[float], Optional[float]]],
) -> List[str]:
    """Return candidate image paths that satisfy all provided metric bounds."""

    def ok(fname: str) -> bool:
        rec = metrics.get(Path(fname).name)
        if rec is None:
            return False
        ch = rec.get("characteristics", {})
        for key, (mn, mx) in bounds.items():
            val = ch.get(key)
            if val is None:
                return False
            if mn is not None and val < mn:
                return False
            if mx is not None and val > mx:
                return False
        return True

    return [c for c in candidates if ok(c)]


def cmd_tuning(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    with open(index_dir / "train_positives.txt", "r", encoding="utf-8") as f:
        pos_all = [ln.strip() for ln in f if ln.strip()]
    with open(index_dir / "train_negatives_bg_only.txt", "r", encoding="utf-8") as f:
        neg_all = [ln.strip() for ln in f if ln.strip()]

    if args.strategy == "random":
        pos_sorted = deterministic_sorted(pos_all)
        neg_sorted = deterministic_sorted(neg_all)
        pos_sel = pos_sorted[: args.n_pos]
        neg_sel = neg_sorted[: args.n_neg]
    else:
        if not args.metrics_file:
            raise SystemExit("metrics strategy requires --metrics-file ...")
        metrics = load_metrics_jsons([Path(p) for p in args.metrics_file])
        pos_bounds = {
            "luma": (args.pos_luma_min, args.pos_luma_max),
            "rms_contrast": (args.pos_contrast_min, args.pos_contrast_max),
            "edge_density": (args.pos_edge_min, args.pos_edge_max),
            "wavelet_mad_db1": (args.pos_noise_min, args.pos_noise_max),
        }
        neg_bounds = {
            "luma": (args.neg_luma_min, args.neg_luma_max),
            "rms_contrast": (args.neg_contrast_min, args.neg_contrast_max),
            "edge_density": (args.neg_edge_min, args.neg_edge_max),
            "wavelet_mad_db1": (args.neg_noise_min, args.neg_noise_max),
        }
        pos_bounds = {
            k: v for k, v in pos_bounds.items() if any(x is not None for x in v)
        }
        neg_bounds = {
            k: v for k, v in neg_bounds.items() if any(x is not None for x in v)
        }
        pos_candidates = (
            filter_by_metrics(pos_all, metrics, pos_bounds)
            if pos_bounds
            else deterministic_sorted(pos_all)
        )
        neg_candidates = (
            filter_by_metrics(neg_all, metrics, neg_bounds)
            if neg_bounds
            else deterministic_sorted(neg_all)
        )
        pos_sel = deterministic_sorted(pos_candidates)[: args.n_pos]
        neg_sel = deterministic_sorted(neg_candidates)[: args.n_neg]

    def write_list(name: str, items: List[str]) -> None:
        path = index_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(it + "\n")
        print(f"  wrote {name}: {len(items)}")

    print("[tuning] Summary:")
    write_list("tune_pos.txt", pos_sel)
    write_list("tune_neg.txt", neg_sel)


# ===============================================================
# Subcommand: emit-yaml — write Ultralytics config files
# ===============================================================


def cmd_emit_yaml(args: argparse.Namespace) -> None:
    ensure_dir(Path(args.out_data).parent)
    ensure_dir(Path(args.out_train).parent)

    data_yaml = f"""
path: .
train: index/train_mixed.txt
val: index/val_internal.txt
test: index/test_all.txt
names:
  0: {YOLO_NAMES[0]}
  1: {YOLO_NAMES[1]}
  2: {YOLO_NAMES[2]}
""".lstrip()

    Path(args.out_data).write_text(data_yaml, encoding="utf-8")

    train_yaml = """
# Repro
seed: 42
amp: true
rect: true

# Training
imgsz: 640
epochs: 100
patience: 20
batch: auto
workers: 8

# Augmentations (clarity-first; keep fixed across all runs)
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.0
mosaic: 0.0
mixup: 0.0
fliplr: 0.5
scale: 0.5
""".lstrip()

    Path(args.out_train).write_text(train_yaml, encoding="utf-8")

    print("[emit-yaml] Wrote:")
    print(f"  {relpath(Path(args.out_data))}")
    print(f"  {relpath(Path(args.out_train))}")


# ===============================================================
# Subcommand: audit — verify label files exist for index lists
# ===============================================================


def cmd_audit(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    raw_label = Path(args.mirror_label)

    def read_list(name: str) -> List[str]:
        p = index_dir / name
        if not p.exists():
            return []
        with open(p, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]

    checks = {
        "train_mixed.txt": read_list("train_mixed.txt"),
        "val_internal.txt": read_list("val_internal.txt"),
        "test_all.txt": read_list("test_all.txt"),
    }

    problems = 0

    def label_path_for_image(img_path: str) -> Path:
        p = Path(img_path)
        parts = p.parts
        split = None
        for i, comp in enumerate(parts):
            if comp == "images" and i + 1 < len(parts):
                split = parts[i + 1]
                break
        if split not in {"train", "val"}:
            split = "train"
        stem = p.with_suffix("").name
        return raw_label / split / f"{stem}.txt"

    for list_name, items in checks.items():
        if not items:
            print(f"[audit] WARN: list missing or empty: {list_name}")
            continue
        missing = []
        for img in items:
            lp = label_path_for_image(img)
            if not lp.exists():
                missing.append((img, relpath(lp)))
        if missing:
            problems += len(missing)
            print(f"[audit] MISSING labels for {list_name}: {len(missing)}")
            for img, expected in missing[:10]:
                print(f"  - {img} -> expected {expected}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
        else:
            print(f"[audit] OK: {list_name} ({len(items)} items)")

    if problems:
        raise SystemExit(f"[audit] FAILED with {problems} missing label files")
    else:
        print("[audit] All checks passed.")


# ===============================================================
# CLI wiring
# ===============================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NightOwls data prep CLI (YOLO labels + indices)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # labels
    pl = sub.add_parser(
        "labels",
        help="Generate YOLO labels (ignores removed) and mirror into raw/labels.",
    )
    pl.add_argument("--raw-ann", type=str, default=str(DEF_RAW_ANN))
    pl.add_argument("--raw-img", type=str, default=str(DEF_RAW_IMG))
    pl.add_argument("--out-label", type=str, default=str(DEF_OUT_LABEL))
    pl.add_argument("--mirror-label", type=str, default=str(DEF_MIRROR_LABEL))
    # NEW: allow custom annotation filenames/paths
    pl.add_argument(
        "--train-ann-file",
        type=str,
        default=None,
        help="Path to train annotations JSON (optional).",
    )
    pl.add_argument(
        "--val-ann-file",
        type=str,
        default=None,
        help="Path to val annotations JSON (optional).",
    )
    pl.set_defaults(func=cmd_labels)

    # indices
    pi = sub.add_parser(
        "indices",
        help="Build pools, 90/10 hash split (per pool), and write index lists + catalog.",
    )
    pi.add_argument("--raw-ann", type=str, default=str(DEF_RAW_ANN))
    pi.add_argument("--raw-img", type=str, default=str(DEF_RAW_IMG))
    pi.add_argument("--index-dir", type=str, default=str(DEF_INDEX_DIR))
    # same custom filename support here
    pi.add_argument(
        "--train-ann-file",
        type=str,
        default=None,
        help="Path to train annotations JSON (optional).",
    )
    pi.add_argument(
        "--val-ann-file",
        type=str,
        default=None,
        help="Path to val annotations JSON (optional).",
    )
    pi.set_defaults(func=cmd_indices)

    # train-mixed
    pm = sub.add_parser(
        "train-mixed", help="Create interleaved training list (e.g., 3:1)."
    )
    pm.add_argument("--index-dir", type=str, default=str(DEF_INDEX_DIR))
    pm.add_argument(
        "--ratio", type=str, default="3:1", help="positives:negatives, e.g. 3:1"
    )
    pm.add_argument(
        "--epoch-mult",
        type=float,
        default=1.0,
        help="scale list length deterministically",
    )
    pm.set_defaults(func=cmd_train_mixed)

    # tuning
    pt = sub.add_parser(
        "tuning", help="Create tuning subsets (random/deterministic or metrics-based)."
    )
    pt.add_argument("--index-dir", type=str, default=str(DEF_INDEX_DIR))
    pt.add_argument("--strategy", choices=["random", "metrics"], default="random")
    pt.add_argument("--n-pos", type=int, default=800)
    pt.add_argument("--n-neg", type=int, default=200)
    pt.add_argument(
        "--metrics-file", action="append", help="Per-image metrics JSON (can repeat)."
    )
    # metric bounds (positives)
    pt.add_argument("--pos-luma-min", type=float, default=None)
    pt.add_argument("--pos-luma-max", type=float, default=None)
    pt.add_argument("--pos-contrast-min", type=float, default=None)
    pt.add_argument("--pos-contrast-max", type=float, default=None)
    pt.add_argument("--pos-edge-min", type=float, default=None)
    pt.add_argument("--pos-edge-max", type=float, default=None)
    pt.add_argument("--pos-noise-min", type=float, default=None)
    pt.add_argument("--pos-noise-max", type=float, default=None)
    # metric bounds (negatives)
    pt.add_argument("--neg-luma-min", type=float, default=None)
    pt.add_argument("--neg-luma-max", type=float, default=None)
    pt.add_argument("--neg-contrast-min", type=float, default=None)
    pt.add_argument("--neg-contrast-max", type=float, default=None)
    pt.add_argument("--neg-edge-min", type=float, default=None)
    pt.add_argument("--neg-edge-max", type=float, default=None)
    pt.add_argument("--neg-noise-min", type=float, default=None)
    pt.add_argument("--neg-noise-max", type=float, default=None)
    pt.set_defaults(func=cmd_tuning)

    # emit-yaml
    py = sub.add_parser(
        "emit-yaml", help="Emit Ultralytics data.yaml and train_baseline.yaml."
    )
    py.add_argument("--out-data", type=str, default=str(DEF_CONFIGS_DIR / "data.yaml"))
    py.add_argument(
        "--out-train", type=str, default=str(DEF_CONFIGS_DIR / "train_baseline.yaml")
    )
    py.set_defaults(func=cmd_emit_yaml)

    # audit
    pa = sub.add_parser(
        "audit", help="Verify that labels exist for images referenced in index lists."
    )
    pa.add_argument("--index-dir", type=str, default=str(DEF_INDEX_DIR))
    pa.add_argument("--mirror-label", type=str, default=str(DEF_MIRROR_LABEL))
    pa.set_defaults(func=cmd_audit)

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
