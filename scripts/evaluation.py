"""COCO evaluation, detector timing loop, and Kaggle submission writing/validation."""

import contextlib
import csv
import io
import math
import time
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import CLASS_NAMES

METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
MAX_DETS = 500   # the competition scores at most 500 detections per image


def load_gt(path: Path) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def coco_evaluate(gt: COCO, detections: list, image_ids, max_dets: int = MAX_DETS, per_class: bool = True) -> dict:
    """COCO bbox metrics; ignored regions are iscrowd=1 so detections inside them are not penalised."""
    if not detections:
        return {**dict.fromkeys(METRIC_KEYS, 0.0), "per_class": {}}
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = COCOeval(gt, gt.loadRes(detections), "bbox")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.params.maxDets = [1, 10, max_dets]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    # pycocotools' own summary hard-codes maxDets=100 for its headline AP (it prints -1 for any other
    # limit), so average the precision table directly at the largest maxDets instead.
    precision = evaluator.eval["precision"]    # [iou, recall, class, area (all, small, medium, large), maxDets]

    def mean_ap(area: int, iou=slice(None), klass=slice(None)):
        values = precision[iou, :, klass, area, -1]
        values = values[values > -1]          # -1 marks "no ground truth for this slice"
        return float(values.mean()) if values.size else None

    result = dict(zip(METRIC_KEYS, [mean_ap(0), mean_ap(0, iou=0), mean_ap(0, iou=5), mean_ap(1), mean_ap(2), mean_ap(3)]))
    result["per_class"] = {}
    if per_class:
        for k, category_id in enumerate(evaluator.params.catIds):
            result["per_class"][CLASS_NAMES[category_id - 1]] = mean_ap(0, klass=k)
    return result


def run_detector(detect, root: Path, split: str, images: list, device: str, log_every: int = 500):
    """Run detect(PIL image) on every frame. Returns (COCO-format detections, mean ms per frame).

    Only the detect() call is timed (image decoding excluded), after 3 warm-up calls, so every model
    is measured the same way at batch size 1.
    """
    import torch

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    with Image.open(root / split / images[0]["file_name"]) as source:
        warm = source.convert("RGB")
    for _ in range(3):
        detect(warm)
    detections, seconds = [], 0.0
    for index, info in enumerate(images, 1):
        with Image.open(root / split / info["file_name"]) as source:
            frame = source.convert("RGB")
        sync()
        start = time.perf_counter()
        found = detect(frame)
        sync()
        seconds += time.perf_counter() - start
        for class_id, score, x1, y1, x2, y2 in found:
            detections.append({"image_id": info["id"], "category_id": int(class_id) + 1,
                               "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
        if log_every and index % log_every == 0:
            print(f"  {index}/{len(images)} frames", flush=True)
    return detections, 1000 * seconds / len(images)


# ---- Kaggle submission --------------------------------------------------------------------------
def prediction_string(found: list, width: float, height: float, max_dets: int = MAX_DETS) -> str:
    """`class score x y w h` groups in original-image pixels, clipped to the frame, best scores first."""
    parts = []
    for class_id, score, x1, y1, x2, y2 in sorted(found, key=lambda d: -d[1])[:max_dets]:
        x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
        y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue   # sub-pixel boxes cannot match anything and print as zero-size
        parts += [str(int(class_id)), f"{score:.5f}", f"{x1:.1f}", f"{y1:.1f}", f"{x2 - x1:.1f}", f"{y2 - y1:.1f}"]
    return " ".join(parts)


def write_submission(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "PredictionString"])
        writer.writerows(rows)


def validate_submission(path: Path, test_csv: Path, max_dets: int = MAX_DETS):
    """Check a submission against test_images.csv. Returns (problems, stats); no problems means valid."""
    with test_csv.open(newline="") as f:
        test = list(csv.DictReader(f))
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)
    problems = []
    if header != ["image_id", "PredictionString"]:
        problems.append(f"header is {header}, expected ['image_id', 'PredictionString']")
    if len(rows) != len(test):
        problems.append(f"{len(rows)} rows, expected {len(test)}")
    counts = []
    for row, expected in zip(rows, test):
        if len(row) != 2 or row[0] != expected["image_id"]:
            problems.append(f"row id {row[:1]} does not match test image {expected['image_id']}")
            continue
        values = row[1].split()
        if len(values) % 6:
            problems.append(f"{row[0]}: {len(values)} values is not a multiple of 6")
            continue
        counts.append(len(values) // 6)
        if counts[-1] > max_dets:
            problems.append(f"{row[0]}: {counts[-1]} detections, limit {max_dets}")
        width, height = float(expected["width"]), float(expected["height"])
        for i in range(0, len(values), 6):
            try:
                class_id = int(values[i])
                score, x, y, w, h = map(float, values[i + 1:i + 6])
            except ValueError:
                problems.append(f"{row[0]}: non-numeric detection {values[i:i + 6]}")
                break
            numbers = (score, x, y, w, h)
            if not (0 <= class_id < len(CLASS_NAMES) and all(map(math.isfinite, numbers)) and 0 <= score <= 1
                    and w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= width + 1 and y + h <= height + 1):
                problems.append(f"{row[0]}: invalid detection {values[i:i + 6]}")
                break
        if len(problems) >= 20:
            problems.append("... stopping after 20 problems")
            break
    stats = {"rows": len(rows), "empty_rows": counts.count(0),
             "detections_per_image": (min(counts), max(counts)) if counts else None}
    return problems, stats
