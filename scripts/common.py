"""Constants and helpers shared by every numbered stage script."""

import json
import random
from collections import defaultdict
from pathlib import Path

SLUG = "multiple-object-video-detection-hw-2-ucu-ai-2026"
SEED = 42
# COCO category ids are 1-based; YOLO labels and the submission use 0-based ids.
CLASS_NAMES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"   # small json/csv files that are committed
FIGURES = REPO / "images"    # figures that may be used in the report
RUNS = REPO / "runs"         # checkpoints and predictions (git-ignored)


def data_root(explicit: Path | None = None) -> Path:
    """Find the competition files: --data, ./data, Kaggle input, or the Colab kagglehub cache."""
    candidates = [explicit] if explicit else []
    candidates += [
        REPO / "data",
        Path("/kaggle/input") / SLUG,
        Path("/kaggle/input/competitions") / SLUG,
        Path("/root/.cache/kagglehub/competitions") / SLUG,
    ]
    for path in candidates:
        if path and (path / "train" / "annotations_coco.json").is_file():
            return Path(path)
    raise FileNotFoundError("Competition data not found; pass --data")


def load_coco(root: Path, split: str) -> dict:
    return json.loads((root / split / "annotations_coco.json").read_text())


def split_annotations(coco: dict):
    """Return (objects, ignored_regions); iscrowd=1 marks regions that scoring ignores."""
    objects = [a for a in coco["annotations"] if not a.get("iscrowd", 0)]
    ignored = [a for a in coco["annotations"] if a.get("iscrowd", 0)]
    return objects, ignored


def index_annotations(coco: dict):
    """Per-image lookups: (objects_by_image_id, ignored_regions_by_image_id)."""
    objects, ignored = defaultdict(list), defaultdict(list)
    for a in coco["annotations"]:
        (ignored if a.get("iscrowd", 0) else objects)[a["image_id"]].append(a)
    return objects, ignored


def sample_frames(images: list, stride: int) -> list:
    """Every `stride`-th frame of each clip (frame ids start at 1), so all clips stay represented."""
    return [i for i in images if (i["frame_id"] - 1) % stride == 0]


def size_bucket(width: float, height: float) -> str:
    """COCO area buckets, used for the AP_small / medium / large breakdown."""
    area = width * height
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def seed_everything(seed: int = SEED) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str | None = None) -> str:
    import torch

    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"
