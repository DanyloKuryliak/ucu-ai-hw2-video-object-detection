"""Stage 07: where does a model fail? Match predictions to ground truth and summarise the errors.

Needs the predictions saved by `05_evaluate.py --save-predictions`. Detections are kept above --score-thr,
detections whose centre lies in an ignored region are dropped (scoring does not penalise them), and each
remaining detection is matched greedily to a ground-truth box at --iou. Writes results/failures/<name>.json
(recall by size and class, class confusions, worst clips) and overlays of the worst frames in
runs/failures/<name>/ (green = found, red = missed, blue = false positive).
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common import CLASS_NAMES, RESULTS, RUNS, data_root, index_annotations, load_coco, save_json, size_bucket


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b: [n, 4] boxes as x1 y1 x2 y2."""
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(bottom_right - top_left, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def xyxy(annotation_boxes) -> np.ndarray:
    boxes = np.array(annotation_boxes, dtype=float).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def inside(centres: np.ndarray, regions: np.ndarray) -> np.ndarray:
    if not len(regions) or not len(centres):
        return np.zeros(len(centres), bool)
    x, y = centres[:, None, 0], centres[:, None, 1]
    return ((x >= regions[None, :, 0]) & (x <= regions[None, :, 2]) & (y >= regions[None, :, 1]) & (y <= regions[None, :, 3])).any(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="the --name used in stage 05")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=6, help="overlays to draw for the worst frames")
    args = parser.parse_args()

    root = data_root(args.data)
    coco = load_coco(root, "val")
    objects, ignored = index_annotations(coco)
    predictions = json.loads((args.predictions or RUNS / "eval" / f"{args.name}_predictions.json").read_text())
    by_image = defaultdict(list)
    for d in predictions:
        if d["score"] >= args.score_thr:
            by_image[d["image_id"]].append(d)
    seen = {d["image_id"] for d in predictions}
    frames = [i for i in coco["images"] if i["id"] in seen]

    size_hit, size_total = Counter(), Counter()
    class_hit, class_total, class_fp = Counter(), Counter(), Counter()
    confusion, per_clip, per_frame = Counter(), defaultdict(Counter), {}
    for info in frames:
        gt, regions = objects[info["id"]], ignored[info["id"]]
        gt_boxes = xyxy([a["bbox"] for a in gt])
        det = sorted(by_image[info["id"]], key=lambda d: -d["score"])
        det_boxes = xyxy([d["bbox"] for d in det])
        if len(det):
            keep = ~inside((det_boxes[:, :2] + det_boxes[:, 2:]) / 2, xyxy([r["bbox"] for r in regions]))
            det, det_boxes = [d for d, k in zip(det, keep) if k], det_boxes[keep]
        matched_gt, matched_det, wrong_class = set(), set(), set()
        if len(gt) and len(det):
            ious = iou_matrix(det_boxes, gt_boxes)
            for i in range(len(det)):
                candidates = [(ious[i, j], j) for j in range(len(gt)) if j not in matched_gt and ious[i, j] >= args.iou]
                if candidates:
                    j = max(candidates)[1]
                    matched_gt.add(j)
                    matched_det.add(i)
                    if det[i]["category_id"] != gt[j]["category_id"]:
                        wrong_class.add(j)
                        confusion[(CLASS_NAMES[gt[j]["category_id"] - 1], CLASS_NAMES[det[i]["category_id"] - 1])] += 1
        for j, a in enumerate(gt):
            found = j in matched_gt and j not in wrong_class
            bucket, name = size_bucket(a["bbox"][2], a["bbox"][3]), CLASS_NAMES[a["category_id"] - 1]
            size_total[bucket] += 1
            class_total[name] += 1
            size_hit[bucket] += found
            class_hit[name] += found
        for i, d in enumerate(det):
            if i not in matched_det:
                class_fp[CLASS_NAMES[d["category_id"] - 1]] += 1
        missed = len(gt) - len(matched_gt) + len(wrong_class)
        false_pos = len(det) - len(matched_det)
        per_clip[info["video"]].update(gt=len(gt), missed=missed, false_positives=false_pos, frames=1)
        per_frame[info["id"]] = (missed + false_pos, missed, false_pos, det, gt, matched_gt, wrong_class, matched_det, info)

    report = {
        "settings": {"score_thr": args.score_thr, "iou": args.iou, "frames": len(frames)},
        "recall_by_size": {k: round(size_hit[k] / size_total[k], 3) for k in size_total},
        "recall_by_class": {k: round(class_hit[k] / class_total[k], 3) for k in class_total},
        "false_positives_by_class": dict(class_fp),
        "top_class_confusions": [{"truth": t, "predicted": p, "count": n} for (t, p), n in confusion.most_common(8)],
        "clips": {c: {"frames": v["frames"], "recall": round(1 - v["missed"] / max(1, v["gt"]), 3), "false_positives_per_frame": round(v["false_positives"] / v["frames"], 1)}
                  for c, v in sorted(per_clip.items())},
    }
    save_json(RESULTS / "failures" / f"{args.name}.json", report)
    out = RUNS / "failures" / args.name
    out.mkdir(parents=True, exist_ok=True)
    for rank, (_, (errors, missed, fp, det, gt, matched_gt, wrong_class, matched_det, info)) in enumerate(
            sorted(per_frame.items(), key=lambda item: -item[1][0])[:args.top], 1):
        with Image.open(root / "val" / info["file_name"]) as source:
            canvas = source.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        line = max(2, canvas.width // 500)
        for j, a in enumerate(gt):
            x, y, w, h = a["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline="lime" if j in matched_gt and j not in wrong_class else "red", width=line)
        for i, d in enumerate(det):
            if i not in matched_det:
                x, y, w, h = d["bbox"]
                draw.rectangle((x, y, x + w, y + h), outline="deepskyblue", width=line)
        canvas.thumbnail((1920, 1080))
        canvas.save(out / f"{rank:02d}_{info['video']}_{info['frame_id']:04d}_missed{missed}_fp{fp}.jpg", quality=85)
    print(json.dumps({k: report[k] for k in ("recall_by_size", "recall_by_class", "top_class_confusions")}, indent=2))
    print("overlays:", out)


if __name__ == "__main__":
    main()
