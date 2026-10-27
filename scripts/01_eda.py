"""Stage 01: exploratory data analysis on the COCO annotations (CPU only, no images needed).

Writes results/eda.json (numbers for the report) and images/eda_*.png (figures).
Optional --preview draws ground truth and ignored regions on frames that exist locally.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from common import CLASS_NAMES, FIGURES, RESULTS, data_root, load_coco, save_json, size_bucket, split_annotations

SPLITS = ("train", "val")
STRIDES = (1, 5, 10, 20)


def quantiles(values):
    return {f"p{q}": round(float(np.percentile(values, q)), 1) for q in (5, 25, 50, 75, 95)}


def summarize(coco: dict) -> dict:
    objects, ignored = split_annotations(coco)
    images = {i["id"]: i for i in coco["images"]}
    sqrt_area = np.array([np.sqrt(a["bbox"][2] * a["bbox"][3]) for a in objects])
    classes = Counter(a["category_id"] - 1 for a in objects)
    buckets = Counter(size_bucket(a["bbox"][2], a["bbox"][3]) for a in objects)
    per_class_size = {
        CLASS_NAMES[c]: quantiles([np.sqrt(a["bbox"][2] * a["bbox"][3]) for a in objects if a["category_id"] - 1 == c])
        for c in sorted(classes)
    }
    per_frame, ignored_per_frame = Counter(), Counter()
    for a in objects:
        per_frame[a["image_id"]] += 1
    for a in ignored:
        ignored_per_frame[a["image_id"]] += 1
    per_clip = defaultdict(lambda: {"frames": 0, "objects": 0, "ignored": 0})
    for image_id, image in images.items():
        clip = per_clip[image["video"]]
        clip["frames"] += 1
        clip["objects"] += per_frame[image_id]
        clip["ignored"] += ignored_per_frame[image_id]
    for clip in per_clip.values():
        clip["objects_per_frame"] = round(clip["objects"] / clip["frames"], 1)
    counts = np.array([per_frame[i] for i in images])
    return {
        "frames": len(images), "clips": len(per_clip), "objects": len(objects), "ignored_regions": len(ignored),
        "frames_without_objects": int((counts == 0).sum()),
        "class_counts": {CLASS_NAMES[c]: n for c, n in sorted(classes.items())},
        "size_bucket_share": {k: round(v / len(objects), 3) for k, v in sorted(buckets.items())},
        "sqrt_area_px": quantiles(sqrt_area), "sqrt_area_px_by_class": per_class_size,
        "objects_per_frame": {**quantiles(counts), "mean": round(float(counts.mean()), 1), "max": int(counts.max())},
        "ignored_per_frame_mean": round(len(ignored) / len(images), 1),
        "image_sizes": {f"{w}x{h}": n for (w, h), n in Counter((i["width"], i["height"]) for i in images.values()).most_common()},
        "per_clip": dict(sorted(per_clip.items())),
    }, sqrt_area, counts


def track_persistence(coco: dict) -> dict:
    """How much of a frame's content is still there k frames later: the argument for the sampling stride."""
    objects, _ = split_annotations(coco)
    images = {i["id"]: i for i in coco["images"]}
    tracks = defaultdict(dict)  # (clip, track) -> {frame: (cx, cy, size)}
    for a in objects:
        image = images[a["image_id"]]
        x, y, w, h = a["bbox"]
        tracks[(image["video"], a["track_id"])][image["frame_id"]] = (x + w / 2, y + h / 2, np.sqrt(w * h))
    result = {}
    for k in STRIDES:
        kept, total, shifts = 0, 0, []
        for frames in tracks.values():
            for frame, (cx, cy, size) in frames.items():
                total += 1
                later = frames.get(frame + k)
                if later:
                    kept += 1
                    shifts.append(np.hypot(later[0] - cx, later[1] - cy) / size)
        result[f"stride_{k}"] = {"object_still_present": round(kept / total, 3),
                                 "median_shift_in_box_sizes": round(float(np.median(shifts)), 3) if shifts else None}
    return result


def plot(summary: dict, sizes: dict, counts: dict):
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(CLASS_NAMES))
    for offset, split in zip((-0.2, 0.2), SPLITS):
        share = np.array([summary[split]["class_counts"].get(n, 0) for n in CLASS_NAMES], dtype=float)
        axes[0].bar(x + offset, share / share.sum(), 0.4, label=split)
    axes[0].set_xticks(x, CLASS_NAMES, rotation=40)
    axes[0].set_yscale("log")
    axes[0].set(title="Class share of objects (log)", ylabel="share")
    axes[0].legend()
    bins = np.logspace(0.3, 2.6, 60)
    for split in SPLITS:
        axes[1].hist(sizes[split], bins=bins, histtype="step", density=True, label=split)
    for edge, name in ((32, "small|medium"), (96, "medium|large")):
        axes[1].axvline(edge, color="gray", ls="--", lw=0.8)
    axes[1].set(xscale="log", title="Object size, sqrt(w*h) in px", xlabel="px")
    axes[1].legend()
    clips = [(c, v["objects_per_frame"]) for c, v in summary["train"]["per_clip"].items()]
    axes[2].hist([v for _, v in clips], bins=20)
    axes[2].set(title="Objects per frame, mean per train clip", xlabel="objects/frame", ylabel="clips")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_overview.png", dpi=130)
    plt.close(fig)


def preview(root: Path, coco: dict, split: str, limit: int = 3):
    """Draw ground truth (green) and ignored regions (red) on frames available locally."""
    objects, ignored = split_annotations(coco)
    by_image = defaultdict(lambda: ([], []))
    for a in objects:
        by_image[a["image_id"]][0].append(a)
    for a in ignored:
        by_image[a["image_id"]][1].append(a)
    shown = 0
    for image in coco["images"]:
        path = root / split / image["file_name"]
        if not path.is_file():
            continue
        with Image.open(path) as source:
            canvas = source.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        gt, ign = by_image[image["id"]]
        for a, colour in [(a, "lime") for a in gt] + [(a, "red") for a in ign]:
            x, y, w, h = a["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline=colour, width=2)
        FIGURES.mkdir(exist_ok=True)
        canvas.save(FIGURES / f"eda_preview_{split}_{image['video']}_{image['frame_id']:04d}.jpg", quality=85)
        print(f"preview: {image['video']} frame {image['frame_id']}: {len(gt)} objects, {len(ign)} ignored regions")
        shown += 1
        if shown == limit:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    root = data_root(args.data)
    summary, sizes, counts, persistence = {}, {}, {}, {}
    cocos = {s: load_coco(root, s) for s in SPLITS}
    for split, coco in cocos.items():
        summary[split], sizes[split], counts[split] = summarize(coco)
        persistence[split] = track_persistence(coco)
    assert not set(summary["train"]["per_clip"]) & set(summary["val"]["per_clip"]), "train/val share a clip"
    with (root / "test_images.csv").open(newline="") as f:
        test = list(csv.DictReader(f))
    summary["test"] = {"frames": len(test), "clips": len({r["clip"] for r in test}),
                       "image_sizes": dict(Counter(f"{r['width']}x{r['height']}" for r in test).most_common())}
    save_json(RESULTS / "eda.json", {"summary": summary, "track_persistence": persistence})
    plot(summary, sizes, counts)
    if args.preview:
        for split, coco in cocos.items():
            preview(root, coco, split)
    for split in SPLITS:
        s = summary[split]
        print(f"{split}: {s['frames']} frames, {s['clips']} clips, {s['objects']:,} objects, {s['ignored_regions']:,} ignored regions")
        print("  classes:", s["class_counts"])
        print("  size buckets:", s["size_bucket_share"], "| sqrt-area:", s["sqrt_area_px"])
        print("  objects/frame:", s["objects_per_frame"], "| image sizes:", s["image_sizes"])
        print("  track persistence:", persistence[split])
    print("test:", summary["test"])


if __name__ == "__main__":
    main()
