#!/usr/bin/env python3
"""
Stratified evaluation by image characteristics (COCO AP + NightOwls MR‑2).

Inputs
------
- GT COCO json (e.g., nightowls_val.json)
- predictions.json (pixel‑space, COCO format)
- characteristics json (your train/val json with per‑image metrics and image_info.id)

Binning
-------
- Explicit edges:        --feature luma --edges 0 20 40 60 80 255
- Quantile (Q-tiles):    --feature luma --quantiles 4
You can repeat --feature ... for multiple characteristics in one run.

Outputs
-------
- CSV: stratified_metrics.csv (one row per {feature, bin})
- (optional) folder with per‑bin filtered {gt,pred}.json for audit

Notes
-----
- Requires pycocotools installed.
- MR‑2 uses your `missrate_nightowls.py` if importable; otherwise MR‑2 is skipped.
- Binning is done on the characteristic *as stored* in your characteristics json.
"""

import json, argparse, math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import copy

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


@dataclass
class BinSpec:
    name: str  # feature name, e.g., "luma"
    mode: str  # "edges" or "quantiles"
    edges: Optional[List[float]] = None
    q: Optional[int] = None


def load_chars(chars_path: Path) -> Dict[int, Dict[str, float]]:
    """
    Returns: mapping image_id -> {feature_name: value, ..., "_file_name": str}
    """
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
        # simple quantile without interpolation complexity
        idx = int(round(k * (len(xs) - 1) / q))
        edges.append(xs[idx])
    edges.append(xs[-1] + 1e-9)  # open upper edge
    # de-duplicate monotone edges (degenerate distributions)
    dedup = [edges[0]]
    for e in edges[1:]:
        if e > dedup[-1]:
            dedup.append(e)
    if len(dedup) < 2:
        dedup = [xs[0], xs[-1] + 1e-9]
    return dedup


def assign_bins(values: Dict[int, float], edges: List[float]) -> Dict[int, int]:
    """
    Given {image_id: value} and sorted edges [e0,e1,...,eN], assign bin index in [0..N-2]
    using half‑open intervals [e_k, e_{k+1}).
    """
    bins = {}
    for iid, v in values.items():
        # binary search could be used; linear is fine for small N
        b = None
        for k in range(len(edges) - 1):
            if edges[k] <= v < edges[k + 1]:
                b = k
                break
        if b is None:
            # put on last bin if numerically equal to last edge (rare)
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
    """
    Write filtered GT and predictions limited to image_ids. Return paths.
    """
    import json

    gt = json.loads(gt_json.read_text(encoding="utf-8"))
    imgs_set = set(map(int, image_ids))

    # keep only the requested images/annotations
    gt_imgs = [im for im in gt["images"] if int(im["id"]) in imgs_set]
    gt_anns = [a for a in gt["annotations"] if int(a["image_id"]) in imgs_set]

    info = gt.get(
        "info", {"description": "patched by stratified_eval", "version": "1.0"}
    )
    licenses = gt.get("licenses", [])
    cats = gt.get("categories", [])

    for a in gt_anns:
        a.setdefault("iscrowd", 0)

    gt_f = {
        "info": info,
        "licenses": licenses,
        "categories": cats,
        "images": gt_imgs,
        "annotations": gt_anns,
    }

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


def make_r50_gt_json(gt_path: Path, out_path: Path, class_id: int = 1, min_h: int = 50, exclude_occluded: bool = True) -> Path:
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
    cat_ids: list[int] | None = None,
    per_class: bool = False,
) -> Dict[str, Any]:
    # Load GT and inject COCO-required fields if missing
    cocoGt = COCO(str(gt_path))
    if "info" not in cocoGt.dataset:
        cocoGt.dataset["info"] = {
            "description": "injected by stratified_eval",
            "version": "1.0",
        }
    for ann in cocoGt.dataset.get("annotations", []):
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0
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
    out = {
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
        # Compute AP per class_id by re-evaluating per-class
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


def run_stratified(
    gt_json: Path,
    preds_json: Path,
    chars_json: Path,
    bin_specs: List[BinSpec],
    out_csv: Path,
    # AP/AR filtering
    ap_cat_ids: Optional[List[int]] = None,
    ap_per_class: bool = False,
    # MR-2 knobs
    mr2_class_id: int = 1,
    mr2_ignore_id: int = 4,
    mr2_min_h: int = 50,
    mr2_exclude_occluded: bool = True,
    # dump control
    dump_filtered: bool = False,
    strata_root: Path | None = None,
):
    # load characteristics
    feats_by_img = load_chars(chars_json)

    rows = []
    for spec in bin_specs:
        # collect values for this feature
        series = {
            iid: feats.get(spec.name)
            for iid, feats in feats_by_img.items()
            if spec.name in feats
        }
        if not series:
            print(
                f"[WARN] Feature '{spec.name}' not found in characteristics JSON; skipping."
            )
            continue

        # choose edges
        if spec.mode == "edges":
            edges = sorted(list(map(float, spec.edges)))
        elif spec.mode == "quantiles":
            edges = compute_quantile_edges(list(series.values()), int(spec.q))
        else:
            raise ValueError("Unknown mode")

        # assign bins
        assignments = assign_bins(series, edges)

        # collect per-bin image ids
        nbins = len(edges) - 1
        for b in range(nbins):
            img_ids = [iid for iid, bb in assignments.items() if bb == b]
            if not img_ids:
                continue

            tag = f"{spec.name}_{b:02d}_{edges[b]:.4g}-{edges[b + 1]:.4g}"

            if dump_filtered:
                out_dir = (Path(strata_root) / f"strata_{tag}") if strata_root else None
                f_gt, f_pr = filter_coco_by_images(
                    gt_json, preds_json, img_ids, out_dir=out_dir, tag=tag
                )
                cm = coco_metrics(
                    f_gt, f_pr, cat_ids=ap_cat_ids, per_class=ap_per_class
                )
                if MR2_AVAILABLE:
                    res = mr2_fn(
                        gt_json=f_gt,
                        pred_json=f_pr,
                        class_id=mr2_class_id,
                        ignore_id=mr2_ignore_id,
                        min_h=mr2_min_h,
                        exclude_occluded=mr2_exclude_occluded,
                    )
                    mr2_val = float(res.get('MR2', float('nan')))
                else:
                    mr2_val = None
            else:
                import tempfile

                with tempfile.TemporaryDirectory() as td:
                    tmp_dir = Path(td)
                    f_gt, f_pr = filter_coco_by_images(
                        gt_json, preds_json, img_ids, out_dir=tmp_dir, tag=tag
                    )
                    cm = coco_metrics(
                        f_gt, f_pr, cat_ids=ap_cat_ids, per_class=ap_per_class
                    )
                    if MR2_AVAILABLE:
                        res = mr2_fn(
                            gt_json=f_gt,
                            pred_json=f_pr,
                            class_id=mr2_class_id,
                            ignore_id=mr2_ignore_id,
                            min_h=mr2_min_h,
                            exclude_occluded=mr2_exclude_occluded,
                        )
                    mr2_val = float(res.get('MR2', float('nan')))
                    else:
                        mr2_val = None

            # Overall row (ALL classes selected for AP/AR)
            rows.append(
                {
                    "feature": spec.name,
                    "bin_index": b,
                    "lo": edges[b],
                    "hi": edges[b + 1],
                    "class_id": "ALL",
                    "n_images": cm["n_images"],
                    "n_anns": cm["n_anns"],
                    "AP": cm["AP"],
                    "AP50": cm["AP50"],
                    "AR1": cm["AR1"],
                    "AR10": cm["AR10"],
                    "AR100": cm["AR100"],
                    "MR2": mr2_val,
                    "tag": tag,
                }
            )

            # Optional: per-class rows
            for pc in cm.get("per_class", []):
                rows.append(
                    {
                        "feature": spec.name,
                        "bin_index": b,
                        "lo": edges[b],
                        "hi": edges[b + 1],
                        "class_id": pc["class_id"],
                        "n_images": cm[
                            "n_images"
                        ],  # images are same for all class rows in this bin
                        "n_anns": pc.get("n_anns", 0),  # per-class GT in this bin
                        "AP": pc["AP"],
                        "AP50": pc["AP50"],
                        "AR1": pc["AR1"],
                        "AR10": pc["AR10"],
                        "AR100": pc["AR100"],
                        "MR2": mr2_val
                        if mr2_class_id == pc["class_id"]
                        else None,  # MR-2 only for its class
                        "tag": f"{tag}__cls{pc['class_id']}",
                    }
                )

    # write CSV
    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
            if rows
            else [
                "feature",
                "bin_index",
                "lo",
                "hi",
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


def parse_args():
    ap = argparse.ArgumentParser(
        description="Stratified evaluation by image characteristics."
    )
    ap.add_argument(
        "--gt",
        required=True,
        type=Path,
        help="COCO GT json for split (e.g., nightowls_val.json)",
    )
    ap.add_argument(
        "--pred", required=True, type=Path, help="predictions.json (pixel coords)"
    )
    ap.add_argument(
        "--chars",
        required=True,
        type=Path,
        help="characteristics json (val.json/train.json)",
    )
    ap.add_argument("--out_csv", type=Path, default=Path("runs/stratified_metrics.csv"))
    ap.add_argument(
        "--strata_out_root",
        type=Path,
        default=None,
        help="If set, write per-bin filtered {gt,pred}.json under this root (strata_<tag>/...).",
    )
    ap.add_argument(
        "--dump_filtered",
        action="store_true",
        help="Save per-bin filtered {gt,pred}.json files for audit.",
    )
    # AP/AR category filtering & per-class reporting
    ap.add_argument(
        "--ap_cat_ids",
        type=str,
        default=None,
        help="Comma-separated category IDs to include in AP/AR (e.g. '1' or '1,2'). If omitted, use all categories.",
    )
    ap.add_argument(
        "--ap_per_class",
        action="store_true",
        help="Also output per-class AP rows (one row per class_id per bin).",
    )
    # MR-2 knobs
    ap.add_argument("--mr2_class_id", type=int, default=1)
    ap.add_argument("--mr2_ignore_id", type=int, default=4)
    ap.add_argument("--mr2_min_h", type=int, default=50)
    ap.add_argument(
        "--mr2_include_occluded",
        action="store_true",
        help="include occluded in MR-2 (default: exclude)",
    )

    # Binning specs (repeatable)
    group = ap.add_argument_group("Binning")
    group.add_argument(
        "--feature",
        action="append",
        dest="features",
        help="feature name to bin on (repeatable)",
    )
    group.add_argument(
        "--edges",
        action="append",
        dest="edges_list",
        nargs="+",
        help="explicit edges for the *last* --feature",
    )
    group.add_argument(
        "--quantiles",
        action="append",
        dest="quantiles_list",
        type=int,
        help="Q for the *last* --feature",
    )

    args = ap.parse_args()

    # Build BinSpec list in the order features were declared
    bin_specs: List[BinSpec] = []
    feats = args.features or []
    edges_list = args.edges_list or []
    quants = args.quantiles_list or []

    # Align edges/quantiles to features (only one mode allowed per feature)
    e_ix = q_ix = 0
    for i, feat in enumerate(feats):
        # pick the next provided mode
        if q_ix < len(quants):
            bin_specs.append(BinSpec(name=feat, mode="quantiles", q=quants[q_ix]))
            q_ix += 1
        elif e_ix < len(edges_list):
            # flatten and cast to float
            edges = [float(x) for x in edges_list[e_ix]]
            bin_specs.append(BinSpec(name=feat, mode="edges", edges=edges))
            e_ix += 1
        else:
            ap.error(
                f"No binning provided for feature '{feat}'. Use --edges ... or --quantiles Q"
            )

    return args, bin_specs


def main():
    args, bin_specs = parse_args()
    # Parse --ap_cat_ids into a list[int] (or None)
    if args.ap_cat_ids:
        args.ap_cat_ids = [int(x) for x in str(args.ap_cat_ids).split(",") if x.strip()]
    else:
        args.ap_cat_ids = None
    run_stratified(
        gt_json=args.gt,
        preds_json=args.pred,
        chars_json=args.chars,
        bin_specs=bin_specs,
        out_csv=args.out_csv,
        ap_cat_ids=args.ap_cat_ids,
        ap_per_class=bool(args.ap_per_class),
        mr2_class_id=args.mr2_class_id,
        mr2_ignore_id=args.mr2_ignore_id,
        mr2_min_h=args.mr2_min_h,
        mr2_exclude_occluded=not args.mr2_include_occluded,
        dump_filtered=args.dump_filtered,
        strata_root=args.strata_out_root,
    )


if __name__ == "__main__":
    main()
