"""Stage 05: score a checkpoint on the validation split, the same way for every model.

Reports COCO AP, AP50, AP75, AP by object size, AP per class, optionally AP per clip, and inference time
per frame (batch size 1, image decoding excluded). Ignored regions are handled by the COCO evaluator.
Saves results/eval/<name>.json and, with --save-predictions, runs/eval/<name>_predictions.json for stage 07.

  python scripts/05_evaluate.py rtdetr runs/rtdetr-l-base/weights/best.pt --name rtdetr-l-base --imgsz 640
  python scripts/05_evaluate.py fasterrcnn runs/frcnn-base/best.pt --name frcnn-base --per-clip --save-predictions
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import RESULTS, RUNS, data_root, load_coco, pick_device, sample_frames, save_json
from detectors import FAMILIES, make_detector
from evaluation import MAX_DETS, coco_evaluate, load_gt, run_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--stride", type=int, default=1, help="evaluate every Nth frame (1 = all 2,846)")
    parser.add_argument("--imgsz", type=int, default=640, help="Ultralytics inference size")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=MAX_DETS)
    parser.add_argument("--per-clip", action="store_true", help="also score every validation clip separately")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()

    root, device = data_root(args.data), pick_device(args.device)
    coco = load_coco(root, "val")
    images = sample_frames(coco["images"], args.stride)
    gt = load_gt(root / "val" / "annotations_coco.json")
    detect = make_detector(args.family, args.checkpoint, device, args.imgsz, args.conf, args.max_det)
    print(f"{args.name}: {len(images)} frames on {device}", flush=True)
    detections, ms = run_detector(detect, root, "val", images, device)
    metrics = coco_evaluate(gt, detections, [i["id"] for i in images], args.max_det)
    result = {"name": args.name, "family": args.family, "checkpoint": str(args.checkpoint), "device": device,
              "frames": len(images), "detections": len(detections), "ms_per_frame": round(ms, 2),
              "settings": {"imgsz": args.imgsz, "conf": args.conf, "max_det": args.max_det}, **metrics}
    if args.per_clip:
        clips = defaultdict(list)
        for info in images:
            clips[info["video"]].append(info["id"])
        result["per_clip"] = {}
        for clip, ids in sorted(clips.items()):
            ids_set = set(ids)
            subset = [d for d in detections if d["image_id"] in ids_set]
            result["per_clip"][clip] = {"frames": len(ids), **coco_evaluate(gt, subset, ids, args.max_det, per_class=False)}
    save_json(RESULTS / "eval" / f"{args.name}.json", result)
    if args.save_predictions:
        target = RUNS / "eval" / f"{args.name}_predictions.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(detections))
        print("predictions:", target)
    print(json.dumps({k: v for k, v in result.items() if k != "per_clip"}, indent=2))


if __name__ == "__main__":
    main()
