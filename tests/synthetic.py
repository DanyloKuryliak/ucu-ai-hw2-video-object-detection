"""Tiny synthetic copy of the competition layout, so every stage can be tested without the 11 GB dataset."""

import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

COLOURS = [(220, 40, 40), (40, 200, 60), (50, 70, 230), (230, 200, 30), (200, 60, 200), (40, 210, 210), (255, 140, 0), (255, 255, 255)]


def make_split(root: Path, split: str, clips: list, frames: int, size: tuple, rng: random.Random, labelled: bool = True):
    images, annotations, rows = [], [], []
    image_id, annotation_id = 0, 0
    for video_id, clip in enumerate(clips, 1):
        (root / split / "images" / clip).mkdir(parents=True, exist_ok=True)
        tracks = [(rng.randrange(1, 9), rng.randrange(10, size[0] - 60), rng.randrange(10, size[1] - 50), rng.randrange(20, 50), rng.randrange(20, 40)) for _ in range(5)]
        for frame in range(1, frames + 1):
            image_id += 1
            canvas = Image.new("RGB", size, (90, 90, 90))
            draw = ImageDraw.Draw(canvas)
            name = f"images/{clip}/{frame:07d}.jpg"
            images.append({"id": image_id, "file_name": name, "width": size[0], "height": size[1],
                           "video_id": video_id, "video": clip, "frame_id": frame})
            rows.append({"image_id": f"{clip}_{frame:07d}", "file_name": f"{split}/{name}", "clip": clip, "frame": frame,
                         "width": size[0], "height": size[1]})
            for track, (cls, x, y, w, h) in enumerate(tracks, 1):
                x += frame * 2
                draw.rectangle((x, y, x + w, y + h), fill=COLOURS[cls - 1])
                annotation_id += 1
                annotations.append({"id": annotation_id, "image_id": image_id, "category_id": cls, "bbox": [x, y, w, h],
                                    "area": w * h, "iscrowd": 0, "track_id": track})
            draw.rectangle((5, size[1] - 40, 70, size[1] - 5), fill=(30, 30, 30))   # an ignored crowd region
            annotation_id += 1
            annotations.append({"id": annotation_id, "image_id": image_id, "category_id": 1, "bbox": [5, size[1] - 40, 65, 35],
                                "area": 65 * 35, "iscrowd": 1, "track_id": -1})
            canvas.save(root / split / name, quality=95)
    if labelled:
        categories = [{"id": i, "name": n} for i, n in enumerate(["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"], 1)]
        (root / split / "annotations_coco.json").write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))
    return rows


def make_dataset(root: Path, seed: int = 0) -> Path:
    rng = random.Random(seed)
    make_split(root, "train", ["clip_a", "clip_b"], 10, (320, 180), rng)
    make_split(root, "val", ["clip_c"], 6, (400, 225), rng)
    rows = make_split(root, "test", ["clip_001", "clip_002"], 3, (320, 180), rng, labelled=False)
    with (root / "test_images.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root
