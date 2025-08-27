# scripts/enhance/enhance_dataset.py
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from enhancers import ZeroDCEEnhancer

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            yield p


def save_png(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", optimize=False)  # truly lossless


def write_yaml(path: Path, ssd_root: Path):
    """
    Emit a YOLO data YAML pointing to SSD-resident folders. You will copy (mirror) the HDD outputs to these SSD paths when needed.
    """
    yaml_text = f"""# Auto-generated
    path: {ssd_root}
    train: enhanced/zerodce/train
    val:   enhanced/zerodce/val
    test:  enhanced/zerodce/val
    names:
    0: pedestrian
    1: bicycledriver
    2: motorbikedriver
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser("Zero-DCE dataset enhancer (PNG out)")
    ap.add_argument(
        "--src-root",
        type=Path,
        required=True,
        help=r"Root of originals, e.g. B:\Projects\...\data\raw\images",
    )
    ap.add_argument(
        "--dst-root",
        type=Path,
        required=True,
        help=r"Where to write enhanced images on HDD, e.g. B:\Projects\...\data\enhanced\zerodce",
    )
    ap.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to Zero-DCE checkpoint (.pth/.pt)",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
    )
    ap.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    ap.add_argument("--half", action="store_true", help="half precision on CUDA")
    ap.add_argument(
        "--tile",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Optional tile size for enormous images",
    )
    ap.add_argument("--yaml-out", type=Path, default=Path("configs/data_zerodce.yaml"))
    ap.add_argument(
        "--ssd-root",
        type=Path,
        default=Path(r"C:\pedestrian_detection_lowlight\data"),
        help="Base path that YAML will use (SSD real folder)",
    )
    ap.add_argument(
        "--manifest", type=Path, default=None, help="Optional manifest path (.json)"
    )
    args = ap.parse_args()

    enhancer = ZeroDCEEnhancer(args.weights, device=args.device, half=args.half)

    stats = {
        "images": 0,
        "splits": {},
        "weights": str(args.weights),
        "tile": args.tile,
        "device": args.device,
        "half": args.half,
        "dst_root": str(args.dst_root),
        "src_root": str(args.src_root),
        "timestamp": datetime.now().isoformat(),
    }

    for split in args.splits:
        src = args.src_root / split
        dst = args.dst_root / split
        count = 0
        files = list(iter_images(src))
        pbar = tqdm(files, desc=f"Enhancing {split}", unit="img")
        for ip in pbar:
            rel = ip.relative_to(src)
            op = dst / rel.with_suffix(".png")  # enforce PNG
            if op.exists():
                count += 1
                continue
            img = Image.open(ip).convert("RGB")
            out = enhancer.enhance_pil(
                img, tile=tuple(args.tile) if args.tile else None
            )
            save_png(out, op)
            count += 1
        stats["splits"][split] = count
        stats["images"] += count

    # YAML that always points to SSD
    write_yaml(args.yaml_out, args.ssd_root)

    # Manifest
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    print("Done. Enhanced PNGs at:", args.dst_root)
    print("YAML written to:", args.yaml_out)
    print("YAML expects SSD base at:", args.ssd_root)


if __name__ == "__main__":
    main()
