"""Stage 09: fuse the detections of several models with weighted box fusion (the "own pipeline" step).

Boxes of the same class that overlap (IoU above --iou) are merged into one box whose coordinates are the
score-weighted mean; its score is the mean score over all models, so a box only one model found is
down-weighted. Models vote, and the fused output keeps at most 500 detections per frame.

  # score the fusion on the validation split (inputs come from 05_evaluate.py --save-predictions)
  python scripts/09_ensemble.py val runs/eval/rtdetr-l-e6_predictions.json runs/eval/nb-rtdetr_predictions.json --name ens-rtdetr2
  # fuse two submission files into a new one, then validate it
  python scripts/09_ensemble.py test submissions/a.csv submissions/b.csv --name ens-rtdetr2
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import REPO, RESULTS, data_root, load_coco, save_json
from evaluation import MAX_DETS, coco_evaluate, load_gt, prediction_string, validate_submission, write_submission


def iou_with(box, boxes):
    x1, y1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    x2, y2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def fuse(per_model: list, iou: float = 0.55, max_dets: int = MAX_DETS):
    """per_model: one list per model of (class, score, x1, y1, x2, y2). Returns the fused list."""
    n_models = len(per_model)
    by_class = defaultdict(list)
    for model, found in enumerate(per_model):
        for class_id, score, *box in found:
            by_class[class_id].append((score, box))
    fused = []
    for class_id, items in by_class.items():
        items.sort(key=lambda t: -t[0])
        members, boxes = [], np.zeros((0, 4))     # members[k] = list of (score, box); boxes[k] = fused box
        for score, box in items:
            k = -1
            if len(boxes):
                overlap = iou_with(np.array(box), boxes)
                k = int(overlap.argmax())
                k = k if overlap[k] > iou else -1
            if k < 0:
                members.append([(score, box)])
                boxes = np.vstack([boxes, box])
            else:
                members[k].append((score, box))
                weights = np.array([s for s, _ in members[k]])
                boxes[k] = (weights[:, None] * np.array([b for _, b in members[k]])).sum(0) / weights.sum()
        for group, box in zip(members, boxes):
            score = sum(sorted((s for s, _ in group), reverse=True)[:n_models]) / n_models
            fused.append((class_id, score, *box))
    fused.sort(key=lambda d: -d[1])
    return fused[:max_dets]


def read_submission(path: Path):
    """image_id -> [(class, score, x1, y1, x2, y2)] from a Kaggle submission file."""
    import csv

    out = {}
    with path.open(newline="") as f:
        for row in list(csv.reader(f))[1:]:
            v = row[1].split()
            out[row[0]] = [(int(v[i]), float(v[i + 1]), float(v[i + 2]), float(v[i + 3]),
                            float(v[i + 2]) + float(v[i + 4]), float(v[i + 3]) + float(v[i + 5])) for i in range(0, len(v), 6)]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["val", "test"])
    parser.add_argument("inputs", nargs="+", type=Path, help="prediction JSONs (val) or submission CSVs (test)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    root = data_root(args.data)

    if args.mode == "val":
        coco = load_coco(root, "val")
        ids = [i["id"] for i in coco["images"]]
        runs = []
        for path in args.inputs:
            grouped = defaultdict(list)
            for d in json.loads(path.read_text()):
                x, y, w, h = d["bbox"]
                grouped[d["image_id"]].append((d["category_id"], d["score"], x, y, x + w, y + h))
            runs.append(grouped)
        detections = []
        for image_id in ids:
            for class_id, score, x1, y1, x2, y2 in fuse([r[image_id] for r in runs], args.iou):
                detections.append({"image_id": image_id, "category_id": class_id, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
        gt = load_gt(root / "val" / "annotations_coco.json")
        metrics = coco_evaluate(gt, detections, ids)
        result = {"name": args.name, "family": "ensemble", "checkpoint": " + ".join(p.stem for p in args.inputs), "device": "mps",
                  "frames": len(ids), "detections": len(detections), "ms_per_frame": round(sum(json.loads((RESULTS / "eval" / f"{p.stem.removesuffix('_predictions')}.json").read_text())["ms_per_frame"] for p in args.inputs), 2), "settings": {"iou": args.iou}, **metrics}
        save_json(RESULTS / "eval" / f"{args.name}.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "per_class"}, indent=2))
    else:
        files = [read_submission(p) for p in args.inputs]
        test = list(__import__("csv").DictReader((root / "test_images.csv").open()))
        rows = [[t["image_id"], prediction_string(fuse([f[t["image_id"]] for f in files], args.iou), float(t["width"]), float(t["height"]))]
                for t in test]
        target = REPO / "submissions" / f"{args.name}.csv"
        write_submission(rows, target)
        problems, stats = validate_submission(target, root / "test_images.csv")
        print(target, stats, "valid" if not problems else problems[:5])


if __name__ == "__main__":
    main()
