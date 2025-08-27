#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import numpy as np


def median_or_none(vals):
    if not vals:
        return None
    return float(np.median(vals))


def is_ped(ann):
    # NightOwls pedestrian is category_id == 1
    return ann.get("category_id") == 1


def is_reasonable(ann):
    # Reasonable: height >= 50 and NOT occluded
    bbox = ann.get("bbox", [0, 0, 0, 0])
    h = float(bbox[3]) if len(bbox) == 4 else 0.0
    occl = ann.get("occluded", False)
    # Some files use null -> treat as not occluded
    return (h >= 50.0) and (occl is not True)


def add_size_fields(chars_path: Path, out_path: Path, inplace: bool = False):
    data = json.loads(Path(chars_path).read_text(encoding="utf-8"))
    changed = 0

    for img_key, rec in data.items():
        anns = rec.get("annotations", [])
        ped_anns = [a for a in anns if is_ped(a)]

        # Heights (all pedestrians)
        heights_all = []
        for a in ped_anns:
            bbox = a.get("bbox", [0, 0, 0, 0])
            if len(bbox) == 4:
                heights_all.append(float(bbox[3]))

        # Heights (Reasonable subset)
        heights_r50 = []
        for a in ped_anns:
            if is_reasonable(a):
                bbox = a.get("bbox", [0, 0, 0, 0])
                if len(bbox) == 4:
                    heights_r50.append(float(bbox[3]))

        ch = rec.setdefault("characteristics", {})
        ch["median_h_all_px"] = median_or_none(heights_all)
        ch["median_h_r50_px"] = median_or_none(heights_r50)
        ch["n_ped_all"] = int(len(heights_all))
        ch["n_ped_r50"] = int(len(heights_r50))
        changed += 1

    out_file = chars_path if inplace else out_path
    if out_file is None:
        raise SystemExit("Specify --out_json or use --inplace to overwrite input.")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Wrote {out_file} (updated {changed} images).")


def main():
    ap = argparse.ArgumentParser(
        description="Inject median pedestrian height (px) into characteristics JSON."
    )
    ap.add_argument(
        "--in_json",
        type=Path,
        required=True,
        help="Path to results/images/{train,val}.json",
    )
    ap.add_argument(
        "--out_json", type=Path, default=None, help="Where to write the updated JSON"
    )
    ap.add_argument(
        "--inplace", action="store_true", help="Overwrite --in_json in place"
    )
    args = ap.parse_args()

    add_size_fields(args.in_json, args.out_json, inplace=args.inplace)


if __name__ == "__main__":
    main()
