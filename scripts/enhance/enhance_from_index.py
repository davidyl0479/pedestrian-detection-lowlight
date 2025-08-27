from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image
from enhancers import ZeroDCEEnhancer  # same folder as this file


def load_list(p: Path) -> list[Path]:
    lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [Path(l) for l in lines]


def main():
    ap = argparse.ArgumentParser("Enhance images listed in index files (write PNGs).")
    ap.add_argument(
        "--indexes",
        nargs="+",
        required=True,
        help="e.g. index/train_mixed.txt index/val_internal.txt",
    )
    ap.add_argument(
        "--base",
        type=Path,
        default=Path("."),
        help="Base dir that index paths are relative to (YAML 'path').",
    )
    ap.add_argument(
        "--dst-base",
        type=Path,
        required=True,
        help=r"Dst base, e.g. B:\...\data (masters on HDD)",
    )
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()

    enhancer = ZeroDCEEnhancer(args.weights, device=args.device, half=args.half)

    # Collect unique source/dest pairs
    src_paths = []
    for idx in args.indexes:
        for rel in load_list(Path(idx)):
            src = (args.base / rel).resolve()

            # --- robust remap: handle backslashes and both "data/raw/images" and "raw/images"
            rel_norm = Path(rel).as_posix()  # normalise to forward slashes
            if "data/raw/images/" in rel_norm:
                rel_norm = rel_norm.replace(
                    "data/raw/images/", "data/enhanced/zerodce/images/"
                )
            elif "raw/images/" in rel_norm:
                rel_norm = rel_norm.replace("raw/images/", "enhanced/zerodce/images/")
            else:
                raise ValueError(f"Index path not under raw/images: {rel}")

            rel_png = Path(rel_norm).with_suffix(".png")

            # Destination under BASE first…
            dst_under_base = (args.base / rel_png).resolve()
            # …then move to HDD masters by swapping roots cleanly
            dst = args.dst_base.resolve() / dst_under_base.relative_to(
                args.base.resolve()
            )

            src_paths.append((src, dst))

    # Deduplicate
    seen = set()
    pairs = []
    for s, d in src_paths:
        key = (str(s), str(d))
        if key not in seen:
            seen.add(key)
            pairs.append((s, d))

        if len(pairs) < 3:
            print(f"[MAP] {src} -> {dst}")

    # Enhance
    from tqdm import tqdm

    for src, dst in tqdm(pairs, desc="Enhancing", unit="img"):
        if dst.exists():
            continue
        img = Image.open(src).convert("RGB")
        out = enhancer.enhance_pil(img)
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst, format="PNG", optimize=False)


if __name__ == "__main__":
    main()
