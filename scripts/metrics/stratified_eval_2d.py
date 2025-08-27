#!/usr/bin/env python3
"""
2D stratified evaluation (Cartesian product of two characteristics).

Example
-------
python stratified_eval_2d.py \
  --gt path/to/nightowls_val.json \
  --pred runs/eval/2025-08-19_val/predictions.json \
  --chars results/images/val.json \
  --feat1 luma --edges1 0 20 40 60 80 255 \
  --feat2 edge_density --quantiles2 3 \
  --out_csv runs/stratified_metrics_val_luma_x_edge.csv

Outputs
-------
- CSV: one row per 2D bin (b1, b2)
- Optional per-bin filtered GT/pred jsons for audit

Notes
-----
- Re-uses COCOeval for AP/AR and (optionally) your missrate_nightowls.MR-2.
- Requires pycocotools installed.
"""

import json, argparse, copy
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# --- Optional MR-2 import (NightOwls) ---
MR2_AVAILABLE = False
try:
    from missrate_nightowls import mr2 as mr2_fn  # adjust PYTHONPATH if needed

    MR2_AVAILABLE = True
except Exception:
    MR2_AVAILABLE = False

# --- COCO ---
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np


def load_chars(chars_path: Path) -> Dict[int, Dict[str, float]]:
    data = json.loads(chars_path.read_text(encoding="utf-8"))
    out = {}
    for fname, payload in data.items():
        img = payload.get("image_info", {})
        img_id = img.get("id")
        feats = dict(payload.get("characteristics", {}))
        feats["_file_name"] = fname
        if img_id is not None:
            out[int(img_id)] = feats
    return out


def compute_quantile_edges(values: List[float], q: int) -> List[float]:
    if not values:
        return []
    xs = sorted(values)
    edges = [xs[0]]
    for k in range(1, q):
        idx = int(round(k * (len(xs) - 1) / q))
        edges.append(xs[idx])
    edges.append(xs[-1] + 1e-9)
    # de-duplicate
    dedup = [edges[0]]
    for e in edges[1:]:
        if e > dedup[-1]:
            dedup.append(e)
    if len(dedup) < 2:
        dedup = [xs[0], xs[-1] + 1e-9]
    return dedup


def assign_bins(values: Dict[int, float], edges: List[float]) -> Dict[int, int]:
    bins = {}
    for iid, v in values.items():
        b = None
        for k in range(len(edges) - 1):
            if edges[k] <= v < edges[k + 1]:
                b = k
                break
        if b is None:
            b = len(edges) - 2
        bins[iid] = b
    return bins


def filter_coco_by_images(
    gt_json: Path,
    pred_json: Path,
    image_ids: List[int],
    out_dir: Optional[Path] = None,
    tag: str = "bin",
) -> Tuple[Path, Path]:
    gt = json.loads(gt_json.read_text(encoding="utf-8"))
    imgs_set = set(map(int, image_ids))

    gt_imgs = [im for im in gt["images"] if int(im["id"]) in imgs_set]
    gt_anns = [a for a in gt["annotations"] if int(a["image_id"]) in imgs_set]
    gt_f = {**gt, "images": gt_imgs, "annotations": gt_anns}

    preds = json.loads(pred_json.read_text(encoding="utf-8"))
    preds_f = [d for d in preds if int(d.get("image_id", -1)) in imgs_set]

    if out_dir is None:
        out_dir = gt_json.parent / f"strata_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = out_dir / f"gt_{tag}.json"
    pr_path = out_dir / f"pred_{tag}.json"
    gt_path.write_text(json.dumps(gt_f), encoding="utf-8")
    pr_path.write_text(json.dumps(preds_f), encoding="utf-8")
    return gt_path, pr_path


def make_r50_gt_json(
    gt_path: Path,
    out_path: Path,
    class_id: int = 1,
    min_h: int = 50,
    exclude_occluded: bool = True,
) -> Path:
    """
    Create a temporary GT JSON where non-Reasonable pedestrians (cat==class_id with h<min_h
    or occluded when exclude_occluded=True) are marked as iscrowd=1 so they don't count
    against AP. This aligns AP with MR-2's Reasonable subset (APR50).
    """
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    anns = data.get("annotations", [])
    for a in anns:
        if "iscrowd" not in a:
            a["iscrowd"] = 0
        if int(a.get("category_id", -1)) != int(class_id):
            continue
        bb = a.get("bbox", [0, 0, 0, 0])
        h = float(bb[3]) if len(bb) == 4 else 0.0
        occl = a.get("occluded")
        if occl is None:
            occl = a.get("attributes", {}).get("occluded")
        occl = bool(occl)
        is_r50 = (h >= float(min_h)) and (True if not exclude_occluded else (not occl))
        if not is_r50:
            a["iscrowd"] = 1
    out_path.write_text(json.dumps(data), encoding="utf-8")
    return out_path


def coco_metrics(
    gt_path: Path,
    pr_path: Path,
    cat_ids: Optional[List[int]] = None,
    per_class: bool = False,
) -> Dict[str, Any]:
    cocoGt = COCO(str(gt_path))

    # Inject missing COCO fields for NightOwls slices
    if "info" not in cocoGt.dataset:
        cocoGt.dataset["info"] = {
            "description": "injected by stratified_eval_2d",
            "version": "1.0",
        }
    # Ensure iscrowd exists (COCOeval expects it)
    for ann in cocoGt.dataset.get("annotations", []):
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0
    # Rebuild indices after edits
    cocoGt.createIndex()

    cocoDt = cocoGt.loadRes(str(pr_path))
    ev = COCOeval(cocoGt, cocoDt, "bbox")
    if cat_ids:
        ev.params.catIds = list(cat_ids)

    # Overall (possibly filtered by catIds)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    stats = ev.stats
    out: Dict[str, Any] = {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
        "AR100": float(stats[8]),
        "n_images": len(cocoGt.getImgIds()),
        "n_anns": len(cocoGt.getAnnIds(catIds=ev.params.catIds))
        if getattr(ev, "params", None) and ev.params.catIds
        else len(cocoGt.getAnnIds()),
        "per_class": [],
        "class_ids": ev.params.catIds
        if ev.params.catIds
        else sorted({c["id"] for c in cocoGt.dataset.get("categories", [])}),
    }

    if per_class:
        class_list = out["class_ids"]
        for cid in class_list:
            ev_c = COCOeval(cocoGt, cocoDt, "bbox")
            ev_c.params = (
                copy.deepcopy(ev.params) if hasattr(ev, "params") else ev_c.params
            )
            ev_c.params.catIds = [cid]
            ev_c.evaluate()
            ev_c.accumulate()
            ev_c.summarize()
            st_c = ev_c.stats
            out["per_class"].append(
                {
                    "class_id": int(cid),
                    "AP": float(st_c[0]),
                    "AP50": float(st_c[1]),
                    "AR1": float(st_c[6]),
                    "AR10": float(st_c[7]),
                    "AR100": float(st_c[8]),
                    "n_anns": len(cocoGt.getAnnIds(catIds=[cid])),
                }
            )

    return out


def main():
    ap = argparse.ArgumentParser(
        description="2D stratified evaluation (feature1 x feature2)."
    )
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--chars", required=True, type=Path)
    ap.add_argument("--feat1", required=True, type=str)
    ap.add_argument("--feat2", required=True, type=str)
    # For feat1
    ap.add_argument("--edges1", type=float, nargs="+")
    ap.add_argument("--quantiles1", type=int)
    # For feat2
    ap.add_argument("--edges2", type=float, nargs="+")
    ap.add_argument("--quantiles2", type=int)
    ap.add_argument(
        "--out_csv", type=Path, default=Path("runs/stratified_metrics_2d.csv")
    )

    # New: AP filtering & per-class reporting & strata root
    ap.add_argument(
        "--ap_cat_ids",
        type=str,
        default=None,
        help="Comma-separated category IDs to include in AP/AR (e.g. '1' or '1,2,3'). If omitted, all cats.",
    )
    ap.add_argument(
        "--ap_per_class",
        action="store_true",
        help="Also output per-class AP rows (one row per class_id per cell).",
    )
    ap.add_argument("--dump_filtered", action="store_true")
    # MR-2 knobs
    ap.add_argument("--mr2_class_id", type=int, default=1)
    ap.add_argument(
        "--strata_out_root",
        type=Path,
        default=None,
        help="If set (and --dump_filtered is used), write per-cell {gt,pred}.json under this root (strata_<tag>/...).",
    )
    ap.add_argument("--mr2_ignore_id", type=int, default=4)
    ap.add_argument("--mr2_min_h", type=int, default=50)
    ap.add_argument("--mr2_include_occluded", action="store_true")
    # APR50 and per-cell feature medians in CSV
    ap.add_argument(
        "--ap_r50",
        action="store_true",
        help="Also compute AP/AP50 on the Reasonable subset (APR50). Uses --mr2_* knobs.",
    )
    ap.add_argument(
        "--summary_features",
        type=str,
        default=None,
        help="Comma-separated feature names to summarize per cell (medians). Adds med_<name> columns.",
    )

    args = ap.parse_args()

    # Parse ap_cat_ids into list[int]
    if args.ap_cat_ids:
        args.ap_cat_ids = [int(x) for x in str(args.ap_cat_ids).split(",") if x.strip()]
    else:
        args.ap_cat_ids = None

    # Parse summary features (e.g., "luma,weber_contrast,laplacian_score")
    summary_feats = (
        [s.strip() for s in str(args.summary_features).split(",")]
        if args.summary_features
        else []
    )

    feats_by_img = load_chars(args.chars)

    # values per feature (numeric only)
    vals1 = {
        iid: float(feats[args.feat1])
        for iid, feats in feats_by_img.items()
        if (args.feat1 in feats) and isinstance(feats[args.feat1], (int, float))
    }
    vals2 = {
        iid: float(feats[args.feat2])
        for iid, feats in feats_by_img.items()
        if (args.feat2 in feats) and isinstance(feats[args.feat2], (int, float))
    }
    common_ids = sorted(set(vals1).intersection(set(vals2)))

    if not common_ids:
        raise SystemExit(f"No images have both features {args.feat1} and {args.feat2}.")

    # edges
    if args.edges1 is not None:
        e1 = sorted(args.edges1)
    elif args.quantiles1 is not None:
        e1 = compute_quantile_edges([vals1[i] for i in common_ids], args.quantiles1)
    else:
        raise SystemExit("Provide --edges1 ... or --quantiles1 for feat1")
    if args.edges2 is not None:
        e2 = sorted(args.edges2)
    elif args.quantiles2 is not None:
        e2 = compute_quantile_edges([vals2[i] for i in common_ids], args.quantiles2)
    else:
        raise SystemExit("Provide --edges2 ... or --quantiles2 for feat2")

    b1 = assign_bins({i: vals1[i] for i in common_ids}, e1)
    b2 = assign_bins({i: vals2[i] for i in common_ids}, e2)

    # group images by (bin1, bin2)
    cells: Dict[Tuple[int, int], List[int]] = {}
    for i in common_ids:
        key = (b1[i], b2[i])
        cells.setdefault(key, []).append(i)

    # evaluate
    import csv

    rows = []
    out_csv = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for (i1, i2), img_ids in sorted(cells.items()):
        lo1, hi1 = e1[i1], e1[i1 + 1]
        lo2, hi2 = e2[i2], e2[i2 + 1]
        tag = f"{args.feat1}_{i1:02d}_{lo1:.4g}-{hi1:.4g}__{args.feat2}_{i2:02d}_{lo2:.4g}-{hi2:.4g}"

        if args.dump_filtered:
            out_dir = (
                (Path(args.strata_out_root) / f"strata_{tag}")
                if args.strata_out_root
                else None
            )
            f_gt, f_pr = filter_coco_by_images(
                args.gt, args.pred, img_ids, out_dir=out_dir, tag=tag
            )
            cm = coco_metrics(
                f_gt, f_pr, cat_ids=args.ap_cat_ids, per_class=bool(args.ap_per_class)
            )
            mr2_val = None
            if MR2_AVAILABLE:
                res = mr2_fn(
                    str(f_gt),
                    str(f_pr),
                    iou_thr=0.5,
                    class_id=args.mr2_class_id,
                    ignore_id=args.mr2_ignore_id,
                    min_h=args.mr2_min_h,
                    exclude_occluded=not args.mr2_include_occluded,
                )
                mr2_val = float(res.get("MR2", float("nan")))
        else:
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                f_gt, f_pr = filter_coco_by_images(
                    args.gt, args.pred, img_ids, out_dir=Path(td), tag=tag
                )
                cm = coco_metrics(
                    f_gt,
                    f_pr,
                    cat_ids=args.ap_cat_ids,
                    per_class=bool(args.ap_per_class),
                )
                mr2_val = None
                if MR2_AVAILABLE:
                    res = mr2_fn(
                        str(f_gt),
                        str(f_pr),
                        iou_thr=0.5,
                        class_id=args.mr2_class_id,
                        ignore_id=args.mr2_ignore_id,
                        min_h=args.mr2_min_h,
                        exclude_occluded=not args.mr2_include_occluded,
                    )
                    mr2_val = float(res.get("MR2", float("nan")))

        # Compute APR50 if requested
        ap_r50, ap50_r50 = None, None
        if args.ap_r50:
            import tempfile

            with tempfile.TemporaryDirectory() as tdr:
                # Re-filter for this cell within the temp dir
                f_gt_r, f_pr_r = filter_coco_by_images(
                    args.gt, args.pred, img_ids, out_dir=Path(tdr), tag=tag
                )
                r50_gt = Path(tdr) / "gt_r50.json"
                make_r50_gt_json(
                    f_gt_r,
                    r50_gt,
                    class_id=args.mr2_class_id,
                    min_h=args.mr2_min_h,
                    exclude_occluded=not args.mr2_include_occluded,
                )
                cm_r50 = coco_metrics(
                    r50_gt, f_pr_r, cat_ids=args.ap_cat_ids, per_class=False
                )
                ap_r50 = float(cm_r50["AP"])
                ap50_r50 = float(cm_r50["AP50"])

        # Per-cell feature medians
        med_cols = {}
        for feat in summary_feats:
            vals = []
            for iid in img_ids:
                v = feats_by_img.get(iid, {}).get(feat)
                if isinstance(v, (int, float)) and not (
                    isinstance(v, float) and (math.isnan(v) or math.isinf(v))
                ):
                    vals.append(float(v))
            med_cols[f"med_{feat}"] = float(np.median(vals)) if len(vals) > 0 else None

        # Overall row (ALL selected cats) for this cell
        row_all = {
            "feat1": args.feat1,
            "bin1": i1,
            "lo1": lo1,
            "hi1": hi1,
            "feat2": args.feat2,
            "bin2": i2,
            "lo2": lo2,
            "hi2": hi2,
            "class_id": "ALL",
            "n_images": cm["n_images"],
            "n_anns": cm["n_anns"],
            "AP": cm["AP"],
            "AP50": cm["AP50"],
            "AR1": cm["AR1"],
            "AR10": cm["AR10"],
            "AR100": cm["AR100"],
            "MR2": mr2_val,
            "AP_R50": ap_r50,
            "AP50_R50": ap50_r50,
            "tag": tag,
        }
        # add per-cell medians to the ALL row so the CSV header includes med_* columns
        row_all.update(med_cols)
        rows.append(row_all)

        # Optional: per-class rows
        for pc in cm.get("per_class", []):
            row_pc = {
                "feat1": args.feat1,
                "bin1": i1,
                "lo1": lo1,
                "hi1": hi1,
                "feat2": args.feat2,
                "bin2": i2,
                "lo2": lo2,
                "hi2": hi2,
                "class_id": pc["class_id"],
                "n_images": cm["n_images"],
                "n_anns": pc.get("n_anns", 0),
                "AP": pc["AP"],
                "AP50": pc["AP50"],
                "AR1": pc["AR1"],
                "AR10": pc["AR10"],
                "AR100": pc["AR100"],
                "MR2": mr2_val if args.mr2_class_id == pc["class_id"] else None,
                "AP_R50": (ap_r50 if args.mr2_class_id == pc["class_id"] else None),
                "AP50_R50": (ap50_r50 if args.mr2_class_id == pc["class_id"] else None),
                "tag": f"{tag}__cls{pc['class_id']}",
            }
            row_pc.update(med_cols)
            rows.append(row_pc)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
            if rows
            else [
                "feat1",
                "bin1",
                "lo1",
                "hi1",
                "feat2",
                "bin2",
                "lo2",
                "hi2",
                "class_id",
                "n_images",
                "n_anns",
                "AP",
                "AP50",
                "AR1",
                "AR10",
                "AR100",
                "MR2",
                "AP_R50",
                "AP50_R50",
                "tag",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] Wrote {out_csv} with {len(rows)} rows.")
    print(f"Binning: {args.feat1} edges={e1} ; {args.feat2} edges={e2}")


if __name__ == "__main__":
    main()
