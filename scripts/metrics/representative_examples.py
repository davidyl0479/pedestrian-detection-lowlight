"""
Representative images per cell/bin for a stratified report.

Given:
- Stratified CSV (1D or 2D)
- GT annotations (COCO/NightOwls)
- Model predictions (COCO detections)
- Image characteristics json (val_size.json)
- Images root directory

For each CSV row (cell/bin), select up to N images:
  1) prototype (closest to cell medians on the binned features),
  2) success exemplar (all R50 GT matched, few/no FPs),
  3) failure-FN exemplar (most unmatched R50 GT),
  4) failure-FP exemplar (most unmatched predictions),
then fill remaining slots with additional prototypes.

Outputs:
- A folder per cell under --out_dir (named by the CSV row 'tag' if present).
- JPEGs with overlays and captions.

Dependencies: pip install pillow pandas numpy pycocotools
"""

import argparse, json, math, os, random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# Utils: IOU, drawing, helpers
# -----------------------------


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(a_x2, b_x2), min(a_y2, b_y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def draw_box(draw: ImageDraw.ImageDraw, box, color, width=3):
    x, y, w, h = box
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)


def put_label(img: Image.Image, lines: List[str], anchor=(5, 5)):
    # Small label box in top-left
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except:
        font = None
    x, y = anchor
    pad = 4
    # Compute text box size
    w = max([draw.textlength(t, font=font) for t in lines]) if lines else 0
    h = sum([(font.getbbox(t)[3] - font.getbbox(t)[1] if font else 12) for t in lines])
    bg = (0, 0, 0, 160)
    # Background rectangle
    draw.rectangle([x, y, x + w + 2 * pad, y + h + 2 * pad], fill=bg)
    ty = y + pad
    for t in lines:
        draw.text((x + pad, ty), t, fill=(255, 255, 255), font=font)
        if font:
            ty += font.getbbox(t)[3] - font.getbbox(t)[1]
        else:
            ty += 12


def clamp_bbox(b, W, H):
    x, y, w, h = b
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(0, min(w, W - x))
    h = max(0, min(h, H - y))
    return [x, y, w, h]


def is_r50(ann, min_h=50, exclude_occluded=True):
    if int(ann.get("category_id", -1)) != 1:
        return False
    bb = ann.get("bbox", [0, 0, 0, 0])
    h = float(bb[3]) if len(bb) == 4 else 0.0
    if h < float(min_h):
        return False
    if exclude_occluded and bool(ann.get("occluded", False)):
        return False
    return True


def safe_float(x):
    try:
        fx = float(x)
        if math.isfinite(fx):
            return fx
    except:
        pass
    return None


# ---------------------------------------
# Build indices for fast per-image access
# ---------------------------------------


def index_gt(gt_json: Path):
    d = json.loads(gt_json.read_text(encoding="utf-8"))
    anns_by_img: Dict[int, List[dict]] = {}
    files_by_img: Dict[int, str] = {}
    for im in d.get("images", []):
        files_by_img[int(im["id"])] = im.get("file_name") or ""
    for a in d.get("annotations", []):
        anns_by_img.setdefault(int(a["image_id"]), []).append(a)
    return anns_by_img, files_by_img


def index_preds(pred_json: Path, cat_id=1, score_thr=0.3):
    preds_by_img: Dict[int, List[dict]] = {}
    preds = json.loads(pred_json.read_text(encoding="utf-8"))
    for det in preds:
        if int(det.get("category_id", -1)) != int(cat_id):
            continue
        if float(det.get("score", 0.0)) < float(score_thr):
            continue
        preds_by_img.setdefault(int(det.get("image_id")), []).append(det)
    # sort by score desc
    for k in preds_by_img:
        preds_by_img[k].sort(key=lambda d: -float(d.get("score", 0.0)))
    return preds_by_img


def load_chars(chars_json: Path):
    data = json.loads(chars_json.read_text(encoding="utf-8"))
    feats_by_img: Dict[int, dict] = {}
    fn_by_img: Dict[int, str] = {}
    for fname, payload in data.items():
        info = payload.get("image_info", {})
        iid = info.get("id")
        ch = dict(payload.get("characteristics", {}))
        if iid is not None:
            feats_by_img[int(iid)] = ch
            # Prefer this filename (often just the filename) to join with images_root
            fn_by_img[int(iid)] = info.get("file_name", fname) or fname
    return feats_by_img, fn_by_img


# -------------------------------------
# Matching TP/FP/FN at IoU >= iou_thr
# -------------------------------------


def match_preds_to_gt(
    preds: List[dict], gts: List[dict], iou_thr=0.5, min_h=50, exclude_occluded=True
):
    # filter GT to pedestrian R50 and non-R50 (for drawing)
    r50 = [a for a in gts if is_r50(a, min_h=min_h, exclude_occluded=exclude_occluded)]
    others = [a for a in gts if int(a.get("category_id", -1)) == 1 and a not in r50]

    # Greedy matching: each GT matched to one prediction
    matched_gt = set()
    tp, fp = [], []
    for det in preds:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(r50):
            if j in matched_gt:
                continue
            iou = iou_xywh(det.get("bbox", [0, 0, 0, 0]), gt.get("bbox", [0, 0, 0, 0]))
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            tp.append((det, r50[best_j], best_iou))
            matched_gt.add(best_j)
        else:
            fp.append((det, None, 0.0))

    fn = [r50[j] for j in range(len(r50)) if j not in matched_gt]
    return tp, fp, fn, r50, others


# ------------------------------------------------------
# Assign images to cells/bins from CSV lo/hi boundaries
# ------------------------------------------------------


def in_interval(x, lo, hi, is_last=False, eps=1e-9):
    # [lo, hi) unless last, then [lo, hi+eps]
    if x is None:
        return False
    if is_last:
        return (x >= lo - eps) and (x <= hi + eps)
    else:
        return (x >= lo - eps) and (x < hi - eps)


def build_cells_from_csv(
    df: pd.DataFrame, feats_by_img: Dict[int, dict]
) -> Dict[int, List[int]]:
    """
    Returns: mapping row_index -> list of image_ids that fall into that row's bin/cell.
    Supports both 1D (feature/bin_index/lo/hi) and 2D (feat1/feat2 pairs).
    """
    cells: Dict[int, List[int]] = {}
    # Precompute list of image ids that have all necessary features
    img_ids = list(feats_by_img.keys())

    for ridx, row in df.iterrows():
        if {"feat1", "bin1", "lo1", "hi1", "feat2", "bin2", "lo2", "hi2"}.issubset(
            row.index
        ):
            f1, lo1, hi1 = str(row["feat1"]), float(row["lo1"]), float(row["hi1"])
            f2, lo2, hi2 = str(row["feat2"]), float(row["lo2"]), float(row["hi2"])
            # Determine if these are last bins (look at max hi per feature across rows)
            max_hi1 = float(df.loc[df["feat1"] == f1, "hi1"].max())
            max_hi2 = float(df.loc[df["feat2"] == f2, "hi2"].max())
            is_last1 = abs(hi1 - max_hi1) <= 1e-9
            is_last2 = abs(hi2 - max_hi2) <= 1e-9

            members = []
            for iid in img_ids:
                ch = feats_by_img.get(iid, {})
                v1 = safe_float(ch.get(f1))
                v2 = safe_float(ch.get(f2))
                if v1 is None or v2 is None:
                    continue
                if in_interval(v1, lo1, hi1, is_last=is_last1) and in_interval(
                    v2, lo2, hi2, is_last=is_last2
                ):
                    members.append(iid)
            cells[ridx] = members

        elif {"feature", "bin_index", "lo", "hi"}.issubset(row.index):
            f, lo, hi = str(row["feature"]), float(row["lo"]), float(row["hi"])
            max_hi = float(df.loc[df["feature"] == f, "hi"].max())
            is_last = abs(hi - max_hi) <= 1e-9

            members = []
            for iid in img_ids:
                ch = feats_by_img.get(iid, {})
                v = safe_float(ch.get(f))
                if v is None:
                    continue
                if in_interval(v, lo, hi, is_last=is_last):
                    members.append(iid)
            cells[ridx] = members

        else:
            # Unknown schema row, skip
            continue

    return cells


# --------------------------------------
# Select representative images per cell
# --------------------------------------


def select_images_for_cell(
    row: pd.Series,
    img_ids: List[int],
    feats_by_img: Dict[int, dict],
    gt_by_img: Dict[int, List[dict]],
    preds_by_img: Dict[int, List[dict]],
    per_cell: int = 3,
    iou_thr: float = 0.5,
    min_h: int = 50,
    exclude_occluded: bool = True,
):
    selections = []  # (iid, role, stats_dict)

    # Decide which features to use for "prototype" closeness
    feat_candidates = []
    if "feat1" in row and isinstance(row["feat1"], str):
        feat_candidates.append(row["feat1"])
    if "feat2" in row and isinstance(row["feat2"], str):
        feat_candidates.append(row["feat2"])
    if not feat_candidates and "feature" in row:
        feat_candidates.append(row["feature"])

    # Target medians from CSV if present
    target = {}
    for f in feat_candidates:
        col = f"med_{f}"
        mv = safe_float(row.get(col))
        # Fallback to bin center if median not present
        if mv is None:
            if "feat1" in row and f == row["feat1"]:
                mv = (float(row["lo1"]) + float(row["hi1"])) / 2.0
            elif "feat2" in row and f == row["feat2"]:
                mv = (float(row["lo2"]) + float(row["hi2"])) / 2.0
            elif "feature" in row and f == row["feature"]:
                mv = (float(row["lo"]) + float(row["hi"])) / 2.0
        if mv is not None:
            target[f] = mv

    # Per-image stats
    stats = {}  # iid -> dict
    for iid in img_ids:
        preds = preds_by_img.get(iid, [])
        gts = gt_by_img.get(iid, [])
        tp, fp, fn, r50, others = match_preds_to_gt(
            preds, gts, iou_thr=iou_thr, min_h=min_h, exclude_occluded=exclude_occluded
        )

        # Closeness to target (sum of absolute diffs over chosen features)
        dist = 0.0
        for f in feat_candidates:
            tv = target.get(f)
            iv = safe_float(feats_by_img.get(iid, {}).get(f))
            if tv is None or iv is None:
                continue
            dist += abs(iv - tv)

        stats[iid] = {
            "tp": len(tp),
            "fp": len(fp),
            "fn": len(fn),
            "n_r50": len(r50),
            "dist": dist,
        }

    # Prototype: min dist, require n_r50 >=1 if possible
    candidates = sorted(
        img_ids, key=lambda i: (stats[i]["n_r50"] <= 0, stats[i]["dist"])
    )
    if candidates:
        selections.append((candidates[0], "prototype", stats[candidates[0]]))

    # Success exemplar: fn==0 and low fp, prefer more R50
    succ = [i for i in img_ids if stats[i]["n_r50"] > 0 and stats[i]["fn"] == 0]
    succ.sort(key=lambda i: (stats[i]["fp"], -stats[i]["n_r50"], stats[i]["dist"]))
    for i in succ:
        if all(s[0] != i for s in selections):
            selections.append((i, "success", stats[i]))
            break

    # Failure-FN: highest fn (tie-break: more R50)
    fail_fn = sorted(
        img_ids, key=lambda i: (-stats[i]["fn"], -stats[i]["n_r50"], stats[i]["dist"])
    )
    for i in fail_fn:
        if stats[i]["fn"] > 0 and all(s[0] != i for s in selections):
            selections.append((i, "failure_fn", stats[i]))
            break

    # Failure-FP: highest fp
    fail_fp = sorted(
        img_ids, key=lambda i: (-stats[i]["fp"], -stats[i]["n_r50"], stats[i]["dist"])
    )
    for i in fail_fp:
        if stats[i]["fp"] > 0 and all(s[0] != i for s in selections):
            selections.append((i, "failure_fp", stats[i]))
            break

    # Fill remaining slots with closest-to-median
    if len(selections) < per_cell:
        pool = [i for i in candidates if all(s[0] != i for s in selections)]
        for i in pool[: max(0, per_cell - len(selections))]:
            selections.append((i, "prototype_extra", stats[i]))

    return selections


# -------------------------
# Main driving entry point
# -------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Export representative images per bin/cell with bbox overlays."
    )
    ap.add_argument(
        "--csv", required=True, type=Path, help="Stratified CSV (1D or 2D)."
    )
    ap.add_argument(
        "--gt", required=True, type=Path, help="GT annotations (NightOwls COCO JSON)."
    )
    ap.add_argument(
        "--pred", required=True, type=Path, help="Predictions (COCO detections)."
    )
    ap.add_argument(
        "--chars",
        required=True,
        type=Path,
        help="Image characteristics JSON (val_size.json).",
    )
    ap.add_argument(
        "--images_root", required=True, type=Path, help="Root folder where images live."
    )
    ap.add_argument(
        "--out_dir", required=True, type=Path, help="Output folder for rendered images."
    )
    ap.add_argument(
        "--per_cell",
        type=int,
        default=3,
        help="How many images per cell/bin to export.",
    )
    ap.add_argument(
        "--score_thr", type=float, default=0.30, help="Score threshold for predictions."
    )
    ap.add_argument(
        "--iou_thr", type=float, default=0.50, help="IoU threshold for TP matching."
    )
    ap.add_argument("--min_h", type=int, default=50, help="Minimum GT height for R50.")
    ap.add_argument(
        "--include_occluded",
        action="store_true",
        help="Include occluded GT in R50 (default: exclude).",
    )
    ap.add_argument(
        "--max_det_per_image",
        type=int,
        default=200,
        help="Optional cap on #preds drawn per image.",
    )
    ap.add_argument(
        "--only_class_id",
        type=int,
        default=1,
        help="Filter predictions to this category id (1=pedestrian).",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    if "class_id" in df.columns:
        df = df[df["class_id"] == "ALL"].copy()

    feats_by_img, filename_by_img = load_chars(args.chars)
    gt_by_img, files_from_gt = index_gt(args.gt)
    preds_by_img = index_preds(
        args.pred, cat_id=args.only_class_id, score_thr=args.score_thr
    )

    # Build membership (row index -> list of image ids)
    cells = build_cells_from_csv(df, feats_by_img)

    # For tagging output folders
    def row_tag(row):
        if "tag" in row and isinstance(row["tag"], str):
            return row["tag"]
        # fallback: build a readable tag
        if {"feat1", "bin1", "lo1", "hi1", "feat2", "bin2", "lo2", "hi2"}.issubset(
            row.index
        ):
            return f"{row['feat1']}_{int(row['bin1']):02d}_{row['lo1']:.3g}-{row['hi1']:.3g}__{row['feat2']}_{int(row['bin2']):02d}_{row['lo2']:.3g}-{row['hi2']:.3g}"
        elif {"feature", "bin_index", "lo", "hi"}.issubset(row.index):
            return f"{row['feature']}_{int(row['bin_index']):02d}_{row['lo']:.3g}-{row['hi']:.3g}"
        else:
            return f"row{int(row.name)}"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ridx, row in df.iterrows():
        img_ids = cells.get(ridx, [])
        if not img_ids:
            continue

        tag = row_tag(row)
        cell_dir = args.out_dir / tag
        cell_dir.mkdir(parents=True, exist_ok=True)

        # Select images for this cell
        sel = select_images_for_cell(
            row=row,
            img_ids=img_ids,
            feats_by_img=feats_by_img,
            gt_by_img=gt_by_img,
            preds_by_img=preds_by_img,
            per_cell=args.per_cell,
            iou_thr=args.iou_thr,
            min_h=args.min_h,
            exclude_occluded=not args.include_occluded,
        )

        # Caption info from CSV if present
        ap_val = row.get("AP", None)
        mr2_val = row.get("MR2", None)

        for iid, role, st in sel:
            # Resolve image path
            # Prefer chars filename (usually just a filename); if empty, try GT mapping
            fn = filename_by_img.get(iid) or files_from_gt.get(iid) or ""
            img_path = args.images_root / fn
            if not img_path.exists():
                # try fallback: maybe file_name in GT is relative path
                img_path = args.images_root / files_from_gt.get(iid, "")
            if not img_path.exists():
                print(f"[WARN] Cannot find image for id {iid} using '{fn}'. Skipping.")
                continue

            # Open image
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            draw = ImageDraw.Draw(img)

            preds = preds_by_img.get(iid, [])[: args.max_det_per_image]
            gts = gt_by_img.get(iid, [])
            tp, fp, fn_list, r50, others = match_preds_to_gt(
                preds,
                gts,
                iou_thr=args.iou_thr,
                min_h=args.min_h,
                exclude_occluded=not args.include_occluded,
            )

            # Draw GT: R50 green, non-R50 gray
            for a in others:
                b = clamp_bbox(a.get("bbox", [0, 0, 0, 0]), W, H)
                draw_box(draw, b, color=(180, 180, 180), width=2)
            for a in r50:
                b = clamp_bbox(a.get("bbox", [0, 0, 0, 0]), W, H)
                draw_box(draw, b, color=(0, 220, 0), width=3)

            # Draw predictions: TP cyan, FP yellow
            for d, g, iou in tp:
                b = clamp_bbox(d.get("bbox", [0, 0, 0, 0]), W, H)
                draw_box(draw, b, color=(0, 200, 255), width=3)
            for d, _, _ in fp:
                b = clamp_bbox(d.get("bbox", [0, 0, 0, 0]), W, H)
                draw_box(draw, b, color=(255, 210, 0), width=3)

            # Build caption
            lines = []
            lines.append(f"cell: {tag}")
            if ap_val is not None:
                lines.append(f"AP={float(ap_val):.3f}")
            if mr2_val is not None and not (
                isinstance(mr2_val, float) and math.isnan(mr2_val)
            ):
                lines.append(f"MR2={float(mr2_val):.3f}")
            lines.append(f"img_id={iid}  role={role}")
            lines.append(
                f"R50 GT={len(r50)}  TP={st['tp']}  FP={st['fp']}  FN={st['fn']}"
            )
            # Add feature readouts for the binned features
            feat_vals = []
            if "feat1" in row:
                f1 = row["feat1"]
                v1 = feats_by_img.get(iid, {}).get(f1, None)
                if v1 is not None:
                    feat_vals.append(f"{f1}={safe_float(v1):.3f}")
            if "feat2" in row:
                f2 = row["feat2"]
                v2 = feats_by_img.get(iid, {}).get(f2, None)
                if v2 is not None:
                    feat_vals.append(f"{f2}={safe_float(v2):.3f}")
            if "feature" in row:
                f = row["feature"]
                v = feats_by_img.get(iid, {}).get(f, None)
                if v is not None:
                    feat_vals.append(f"{f}={safe_float(v):.3f}")
            if feat_vals:
                lines.append("  ".join(feat_vals))

            put_label(img, lines, anchor=(6, 6))

            # Save
            out_name = f"{role}_img{iid}.jpg"
            img.save(str(cell_dir / out_name), quality=92)

    print(f"[OK] Exported representative images to {args.out_dir}")


if __name__ == "__main__":
    main()
