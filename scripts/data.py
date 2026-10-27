"""Datasets and format conversion: COCO json -> TorchVision tensors or an Ultralytics (YOLO) folder."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from common import CLASS_NAMES, index_annotations, load_coco, sample_frames, save_json

GREY = 114   # same value Ultralytics uses for letterbox padding


def mask_regions(image: Image.Image, regions: list, keep: list) -> Image.Image:
    """Paint ignored regions grey so unlabelled crowds are not learned as background.

    `regions` and `keep` are COCO annotations; pixels inside labelled boxes (`keep`) are restored
    in case a labelled object overlaps an ignored region.
    """
    if not regions:
        return image
    original = np.asarray(image)
    pixels = original.copy()
    height, width = pixels.shape[:2]

    def clip(box):
        x, y, w, h = box
        return max(0, int(x)), max(0, int(y)), min(width, int(np.ceil(x + w))), min(height, int(np.ceil(y + h)))

    for region in regions:
        x1, y1, x2, y2 = clip(region["bbox"])
        pixels[y1:y2, x1:x2] = GREY
    for annotation in keep:
        x1, y1, x2, y2 = clip(annotation["bbox"])
        pixels[y1:y2, x1:x2] = original[y1:y2, x1:x2]
    return Image.fromarray(pixels)


class DetectionDataset(Dataset):
    """Frames with at least one labelled object, as (image tensor 0-1, target dict) for TorchVision.

    label_offset: 0 keeps COCO ids 1-8 (Faster R-CNN, id 0 = background); -1 gives 0-7 (RetinaNet).
    """

    def __init__(self, root: Path, split: str, stride: int = 1, label_offset: int = 0,
                 mask_ignored: bool = False, flip: bool = False):
        self.root, self.split = Path(root), split
        self.label_offset, self.mask_ignored, self.flip = label_offset, mask_ignored, flip
        coco = load_coco(self.root, split)
        self.objects, self.ignored = index_annotations(coco)
        self.images = [i for i in sample_frames(coco["images"], stride) if self.objects[i["id"]]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        with Image.open(self.root / self.split / info["file_name"]) as source:
            image = source.convert("RGB")
        if self.mask_ignored:
            image = mask_regions(image, self.ignored[info["id"]], self.objects[info["id"]])
        boxes, labels = [], []
        for annotation in self.objects[info["id"]]:
            x, y, w, h = annotation["bbox"]
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(annotation["category_id"] + self.label_offset)
        tensor = pil_to_tensor(image).float().div_(255)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.flip and random.random() < 0.5:
            tensor = tensor.flip(-1)
            boxes = torch.stack([tensor.shape[-1] - boxes[:, 2], boxes[:, 1], tensor.shape[-1] - boxes[:, 0], boxes[:, 3]], 1)
        return tensor, {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64)}


def collate(batch):
    return tuple(zip(*batch))


def export_yolo(root: Path, out: Path, stride: int = 5, val_stride: int = 1, mask_ignored: bool = False) -> dict:
    """Write an Ultralytics dataset: images (symlinks, or masked copies), labels, data.yaml.

    Ignored regions have no YOLO equivalent, so with mask_ignored=True they are painted grey in the
    TRAIN images. Validation images are never modified because the real test set has no ignore info.
    """
    out = Path(out)
    stats = {}
    for split, step in (("train", stride), ("val", val_stride)):
        coco = load_coco(root, split)
        objects, ignored = index_annotations(coco)
        frames = sample_frames(coco["images"], step)
        for info in frames:
            relative = Path(info["file_name"]).relative_to("images")
            image_path = out / "images" / split / relative
            label_path = out / "labels" / split / relative.with_suffix(".txt")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if image_path.is_symlink() or image_path.exists():
                image_path.unlink()
            source = (root / split / info["file_name"]).resolve()
            if mask_ignored and split == "train" and ignored[info["id"]]:
                with Image.open(source) as original:
                    mask_regions(original.convert("RGB"), ignored[info["id"]], objects[info["id"]]).save(image_path, quality=95)
            else:
                image_path.symlink_to(source)
            width, height = info["width"], info["height"]
            lines = []
            for a in objects[info["id"]]:
                x, y, w, h = a["bbox"]
                x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
                if x2 > x1 and y2 > y1:
                    lines.append(f"{a['category_id'] - 1} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                                 f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        stats[split] = {"frames": len(frames), "clips": len({i["video"] for i in frames}),
                        "objects": sum(len(objects[i["id"]]) for i in frames)}
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}")
    stats["masked_ignored_regions"] = mask_ignored
    save_json(out / "export_stats.json", stats)
    return stats
