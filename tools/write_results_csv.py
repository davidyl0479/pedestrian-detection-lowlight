import os, json, csv
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

eval_dir = Path(os.environ["EVAL_DIR"])
man = json.loads((eval_dir/"manifest.json").read_text())
pred_json = Path(man["predictions_json"])

# internal-val lives under TRAIN → use TRAIN GT json
gt_json = Path("data/raw/annotations/train/nightowls_training.json")

cocoGt = COCO(str(gt_json))
cocoGt.dataset.setdefault("info", {})
for ann in cocoGt.dataset.get("annotations", []):
    ann.setdefault("iscrowd", 0)
cocoGt.createIndex()

cocoDt = cocoGt.loadRes(str(pred_json))
ev = COCOeval(cocoGt, cocoDt, "bbox")
ev.evaluate(); ev.accumulate(); ev.summarize()
s = ev.stats  # [AP50-95, AP50, AP75, AP_S, AP_M, AP_L, AR_1, AR_10, AR_100, AR_S, AR_M, AR_L]

out = eval_dir/"val"/"results.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["AP50-95","AP50","AP75","AP_S","AP_M","AP_L","AR_1","AR_10","AR_100","AR_S","AR_M","AR_L"])
    w.writeheader()
    w.writerow({"AP50-95":s[0],"AP50":s[1],"AP75":s[2],"AP_S":s[3],"AP_M":s[4],"AP_L":s[5],
                "AR_1":s[6],"AR_10":s[7],"AR_100":s[8],"AR_S":s[9],"AR_M":s[10],"AR_L":s[11]})
print("Wrote", out)
