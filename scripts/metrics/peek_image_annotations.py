"""
Peek NightOwls annotations for a single image.

Usage examples:
  # By image id
  python scripts/metrics/peek_image_annotations.py \
    --gt data/raw/annotations/val/nightowls_validation.json \
    --img_id 7024673

  # By file name (exact match)
  python scripts/metrics/peek_image_annotations.py \
    --gt data/raw/annotations/val/nightowls_validation.json \
    --file_name 58c5832bbc26013700159be6.png \
    --show_ignore
"""

import argparse
import json
from pathlib import Path


def load_gt(gt_path: Path):
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    images = {int(im["id"]): im for im in data.get("images", [])}
    images_by_name = {
        im.get("file_name", ""): int(im["id"]) for im in data.get("images", [])
    }
    anns_by_img = {}
    for a in data.get("annotations", []):
        anns_by_img.setdefault(int(a["image_id"]), []).append(a)
    # category id -> name map (fallback if not present in ann)
    cat_name = {}
    for c in data.get("categories", []):
        cid = int(c.get("id", -1))
        if cid >= 0:
            cat_name[cid] = c.get("name") or c.get("supercategory") or f"cat_{cid}"
    return images, images_by_name, anns_by_img, cat_name


def is_ped(a) -> bool:
    return int(a.get("category_id", -1)) == 1


def is_ignore(a) -> bool:
    # NightOwls uses category_id==4 for ignore regions
    return int(a.get("category_id", -1)) == 4 or int(a.get("ignore", 0)) == 1


def bbox_h(a) -> float:
    bb = a.get("bbox", [0, 0, 0, 0])
    try:
        return float(bb[3])
    except Exception:
        return 0.0


def flag(val) -> str:
    return "Y" if val else "N"


def main():
    ap = argparse.ArgumentParser(
        description="Show pedestrians, sizes, occlusion (and ignore) for one NightOwls image."
    )
    ap.add_argument("--gt", required=True, type=Path, help="Path to NightOwls GT JSON.")
    ap.add_argument("--img_id", type=int, help="Image id to inspect.")
    ap.add_argument(
        "--file_name", type=str, help="Image file_name to inspect (exact match)."
    )
    ap.add_argument(
        "--min_h",
        type=int,
        default=50,
        help="Minimum height for Reasonable (kept for portability).",
    )
    ap.add_argument(
        "--include_occluded",
        action="store_true",
        help="If set, occluded pedestrians also count as R50.",
    )
    ap.add_argument(
        "--show_ignore",
        action="store_true",
        help="Also list ignore regions (cat_id==4).",
    )
    args = ap.parse_args()

    if not args.img_id and not args.file_name:
        ap.error("Provide either --img_id or --file_name.")

    images, images_by_name, anns_by_img, cat_name = load_gt(args.gt)

    # resolve image id
    if args.img_id:
        iid = int(args.img_id)
        if iid not in images:
            raise SystemExit(f"Image id {iid} not found in {args.gt}")
    else:
        fn = args.file_name
        if fn not in images_by_name:
            raise SystemExit(f"file_name '{fn}' not found in {args.gt}")
        iid = images_by_name[fn]

    im = images[iid]
    im_anns = anns_by_img.get(iid, [])

    # header
    print("=" * 80)
    print(
        f"Image: id={iid}  file_name='{im.get('file_name', '')}'  size={im.get('width', '?')}x{im.get('height', '?')}"
    )
    print("=" * 80)

    # pedestrians
    peds = [a for a in im_anns if is_ped(a)]
    if peds:
        print("PEDESTRIANS:")
        print(
            f"{'ann_id':>8}  {'h(px)':>6}  {'occl':>4}  {'trunc':>5}  {'diff':>4}  {'ignore':>6}  {'R50?':>5}  bbox [x y w h]"
        )
        for a in peds:
            ann_id = int(a.get("id", -1))
            h = bbox_h(a)
            occl = bool(a.get("occluded", False))
            trunc = bool(a.get("truncated", False))
            diff = bool(a.get("difficult", False))
            ign = bool(a.get("ignore", 0)) or is_ignore(a)
            # Reasonable: ped & h>=min_h & (not occluded unless include_occluded)
            r50 = (h >= args.min_h) and (True if args.include_occluded else (not occl))
            bb = a.get("bbox", [0, 0, 0, 0])
            print(
                f"{ann_id:8d}  {h:6.1f}  {flag(occl):>4}  {flag(trunc):>5}  {flag(diff):>4}  {flag(ign):>6}  {flag(r50):>5}  [{bb[0]:.0f} {bb[1]:.0f} {bb[2]:.0f} {bb[3]:.0f}]"
            )
    else:
        print("No pedestrian annotations in this image.")

    # ignore regions
    if args.show_ignore:
        ig = [a for a in im_anns if is_ignore(a)]
        print("\nIGNORE REGIONS (cat_id==4):" if ig else "\nNo ignore regions.")
        for a in ig:
            ann_id = int(a.get("id", -1))
            bb = a.get("bbox", [0, 0, 0, 0])
            print(
                f"  ann_id={ann_id}  bbox=[{bb[0]:.0f} {bb[1]:.0f} {bb[2]:.0f} {bb[3]:.0f}]"
            )

    print("=" * 80)


if __name__ == "__main__":
    main()
