# scripts/metrics/missrate_nightowls.py
import json, math, argparse
from collections import defaultdict


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(gt_path, class_id=1, ignore_id=4, min_h=50, exclude_occluded=True):
    data = json.loads(open(gt_path, "r", encoding="utf-8").read())
    imgs = {im["id"]: im for im in data["images"]}

    # group GT pedestrians and ignore boxes per image
    gts = defaultdict(list)
    ignores = defaultdict(list)
    npos = 0

    for ann in data["annotations"]:
        cid = ann.get("category_id", None)
        if cid == class_id:
            h = ann["bbox"][3]
            if h < min_h:
                continue
            if exclude_occluded:
                # occlusion flag may live in different places; try common patterns
                occ = ann.get("occluded")
                if occ is None:
                    occ = ann.get("attributes", {}).get("occluded")
                if bool(occ):
                    continue
            gts[ann["image_id"]].append({"bbox": ann["bbox"], "used": False})
            npos += 1
        elif ignore_id is not None and cid == ignore_id:
            ignores[ann["image_id"]].append(ann["bbox"])

    img_ids = list(imgs.keys())
    return img_ids, gts, ignores, npos


def load_preds(pred_path, class_id=1):
    preds_by_img = defaultdict(list)
    with open(pred_path, "r", encoding="utf-8") as f:
        dets = json.loads(f.read())
    for d in dets:
        if d.get("category_id") != class_id:
            continue
        preds_by_img[d["image_id"]].append(
            {"bbox": d["bbox"], "score": float(d["score"])}
        )
    # sort each image's detections by descending score
    for k in preds_by_img:
        preds_by_img[k].sort(key=lambda z: z["score"], reverse=True)
    return preds_by_img


def mr2(
    gt_json,
    pred_json,
    iou_thr=0.5,
    class_id=1,
    ignore_id=4,
    min_h=50,
    exclude_occluded=True,
):
    img_ids, gts, ignores, npos = load_gt(
        gt_json, class_id, ignore_id, min_h, exclude_occluded
    )
    preds = load_preds(pred_json, class_id)

    scores, tps, fps = [], [], []
    nimg = len(img_ids)

    for img_id in img_ids:
        gt_list = gts.get(img_id, [])
        ign_list = ignores.get(img_id, [])
        for det in preds.get(img_id, []):
            bb = det["bbox"]
            sc = det["score"]

            # match to GT (greedy, one-to-one)
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gt_list):
                if g["used"]:
                    continue
                i = iou_xywh(bb, g["bbox"])
                if i > best_iou:
                    best_iou, best_j = i, j

            if best_iou >= iou_thr:
                gt_list[best_j]["used"] = True
                scores.append(sc)
                tps.append(1)
                fps.append(0)
                continue

            # if not TP, check ignore overlap: if overlaps ignore, drop
            ignore_hit = any(iou_xywh(bb, ib) >= iou_thr for ib in ign_list)
            if ignore_hit:
                continue  # neither TP nor FP

            # count as FP
            scores.append(sc)
            tps.append(0)
            fps.append(1)

    if npos == 0:
        return {"MR2": 1.0, "npos": 0, "nimg": nimg, "ntp": 0, "nfp": sum(fps)}

    # sort all detections by score descending
    order = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    tp_cum, fp_cum = [], []
    tp_sum = fp_sum = 0
    for k in order:
        tp_sum += tps[k]
        fp_sum += fps[k]
        tp_cum.append(tp_sum)
        fp_cum.append(fp_sum)

    # compute recall and FPPI at each threshold
    recalls = [tp / npos for tp in tp_cum]
    fppis = [fp / nimg for fp in fp_cum]

    # MR-2 over 9 log-spaced FPPI points in [1e-2, 1e0]
    ref = [
        10**x for x in [(-2 + i * (2.0 / 8)) for i in range(9)]
    ]  # [-2, -1.75, ..., 0]
    mr_points = []
    for r in ref:
        # best recall with FPPI <= r
        rcall = 0.0
        for R, F in zip(recalls, fppis):
            if F <= r and R > rcall:
                rcall = R
        mr_points.append(max(1.0e-10, 1.0 - rcall))
    lamr = math.exp(sum(math.log(m) for m in mr_points) / len(mr_points))

    return {
        "MR2": lamr,
        "npos": npos,
        "nimg": nimg,
        "ntp": tp_cum[-1] if tp_cum else 0,
        "nfp": fp_cum[-1] if fp_cum else 0,
        "iou": iou_thr,
        "min_h": min_h,
        "exclude_occluded": bool(exclude_occluded),
    }


def main():
    ap = argparse.ArgumentParser("NightOwls MR-2 evaluator (pedestrian)")
    ap.add_argument("--gt", required=True, help="GT COCO JSON (NightOwls val JSON)")
    ap.add_argument("--pred", required=True, help="predictions.json (COCO dets)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument(
        "--class_id", type=int, default=1, help="pedestrian category id (NightOwls=1)"
    )
    ap.add_argument(
        "--ignore_id", type=int, default=4, help="ignore category id (NightOwls=4)"
    )
    ap.add_argument("--min_height", type=int, default=50, help="Reasonable: h>=50")
    ap.add_argument(
        "--include_occluded", action="store_true", help="include occluded pedestrians"
    )
    args = ap.parse_args()

    res = mr2(
        args.gt,
        args.pred,
        iou_thr=args.iou,
        class_id=args.class_id,
        ignore_id=args.ignore_id,
        min_h=args.min_height,
        exclude_occluded=not args.include_occluded,
    )
    print(
        f"MR-2={res['MR2'] * 100:.2f}% | IoU={res['iou']} | "
        f"h>={res['min_h']} | exclude_occluded={res['exclude_occluded']} | "
        f"GT pos={res['npos']} | images={res['nimg']} | TP={res['ntp']} FP={res['nfp']}"
    )


if __name__ == "__main__":
    main()
