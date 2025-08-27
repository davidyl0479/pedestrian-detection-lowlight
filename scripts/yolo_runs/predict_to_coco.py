# tools/predict_to_coco.py
import json, argparse
from pathlib import Path
from collections import Counter
import yaml
from ultralytics import YOLO


def norm(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("_", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml")
    ap.add_argument("--gt", required=True, help="GT COCO json for this split")
    ap.add_argument("--model", required=True, help="path to .pt")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument(
        "--outdir", required=True, help="output folder for predictions.json"
    )
    ap.add_argument("--workers", type=int, default=0)  # Windows-friendly
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load data.yaml ---
    with open(args.data, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    base = Path(data_cfg.get("path", "")).resolve()
    split_spec = data_cfg.get(args.split) or data_cfg.get("val")

    # Resolve predict() source (dir / file / .txt list)
    src_path = Path(split_spec)
    if not src_path.exists():
        src_path = (base / split_spec).resolve()
    if not src_path.exists():
        raise FileNotFoundError(
            f"Could not resolve source from data.yaml: {split_spec} (tried {src_path})"
        )

    # --- Load GT and build image + category maps ---
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    im_base_map = {
        Path(im["file_name"]).name: (im["id"], im["width"], im["height"])
        for im in gt["images"]
    }

    names = data_cfg.get("names")
    idx_to_name = (
        {int(k): v for k, v in names.items()}
        if isinstance(names, dict)
        else dict(enumerate(names))
    )
    name_to_catid = {norm(c["name"]): c["id"] for c in gt["categories"]}
    cat_map = {}
    missing = []
    for i, nm in idx_to_name.items():
        key = norm(nm)
        if key in name_to_catid:
            cat_map[i] = name_to_catid[key]
        else:
            missing.append((i, nm))
    if missing:
        raise KeyError(f"No GT category for YOLO classes: {missing}")
    print(f"[predict_to_coco] cat_map: {cat_map}")

    # --- Run predict() streaming ---
    model = YOLO(args.model)
    gen = model.predict(
        source=str(src_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        stream=True,
        verbose=False,
        workers=args.workers,
    )

    # --- Build official COCO dets ---
    official = []
    per_cat = Counter()
    n_imgs = 0
    for r in gen:
        base_name = Path(r.path).name
        rec = im_base_map.get(base_name)
        if rec is None:
            # not in this split's GT; skip
            continue
        img_id, W, H = rec
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue

        # Torch → numpy (or already numpy)
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
        except Exception:
            import numpy as np

            xyxy = np.asarray(boxes.xyxy)
            clss = np.asarray(boxes.cls).astype(int)
            confs = np.asarray(boxes.conf)

        for (x1, y1, x2, y2), c, s in zip(xyxy, clss, confs):
            cid = cat_map.get(int(c))
            if cid is None:
                continue
            w = float(x2 - x1)
            h = float(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            official.append(
                {
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": [float(x1), float(y1), w, h],  # COCO xywh in pixels
                    "score": float(s),
                }
            )
            per_cat[cid] += 1
        n_imgs += 1

    # --- Save ---
    out_json = outdir / "predictions.json"
    out_json.write_text(json.dumps(official), encoding="utf-8")
    print(
        f"[predict_to_coco] wrote {len(official)} dets from {n_imgs} images → {out_json}"
    )


if __name__ == "__main__":
    main()
