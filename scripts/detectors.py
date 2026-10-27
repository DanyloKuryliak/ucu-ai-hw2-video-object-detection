"""Model construction (with the tuning knobs), checkpoints, and a uniform detector interface."""

from functools import partial
from pathlib import Path

import torch
from torch import nn
from torchvision.models.detection import (fasterrcnn_resnet50_fpn, retinanet, retinanet_resnet50_fpn_v2)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.ops import sigmoid_focal_loss
from torchvision.transforms.functional import pil_to_tensor

TORCHVISION_FAMILIES = ("fasterrcnn", "retinanet")
FAMILIES = TORCHVISION_FAMILIES + ("yolo", "rtdetr")
NUM_CLASSES = 8

# Defaults match TorchVision except where noted; every key can be overridden from the training CLI.
DEFAULTS = {
    "min_size": 800, "max_size": 1333,
    # Faster R-CNN: proposal stage and RoI sampling
    "rpn_pre_nms_top_n_train": 2000, "rpn_post_nms_top_n_train": 2000,
    "box_batch_size_per_image": 512, "box_positive_fraction": 0.25,
    # RetinaNet: anchors (base size per FPN level, 3 scales each), assignment thresholds, losses
    "anchor_base": [32, 64, 128, 256, 512], "aspect_ratios": [0.5, 1.0, 2.0],
    "fg_iou": 0.5, "bg_iou": 0.4, "focal_alpha": 0.25, "focal_gamma": 2.0, "box_loss": "l1",
}


def set_focal_loss(alpha: float, gamma: float) -> None:
    """TorchVision hard-codes alpha/gamma inside RetinaNet's loss, so rebind the function it calls."""
    retinanet.sigmoid_focal_loss = partial(sigmoid_focal_loss, alpha=alpha, gamma=gamma)


def anchor_generator(base: list, ratios: list) -> AnchorGenerator:
    sizes = tuple((int(b), int(b * 2 ** (1 / 3)), int(b * 2 ** (2 / 3))) for b in base)
    return AnchorGenerator(sizes, (tuple(ratios),) * len(sizes))


def build_torchvision(family: str, cfg: dict | None = None, pretrained: bool = True) -> nn.Module:
    cfg = {**DEFAULTS, **(cfg or {})}
    weights = "DEFAULT" if pretrained else None
    if family == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(
            weights=weights, weights_backbone=None, min_size=cfg["min_size"], max_size=cfg["max_size"],
            rpn_pre_nms_top_n_train=cfg["rpn_pre_nms_top_n_train"], rpn_post_nms_top_n_train=cfg["rpn_post_nms_top_n_train"],
            box_batch_size_per_image=cfg["box_batch_size_per_image"], box_positive_fraction=cfg["box_positive_fraction"],
            box_score_thresh=0.001, box_detections_per_img=500)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES + 1)
        return model
    if family != "retinanet":
        raise ValueError(f"not a TorchVision family: {family}")
    model = retinanet_resnet50_fpn_v2(
        weights=weights, weights_backbone=None, min_size=cfg["min_size"], max_size=cfg["max_size"],
        fg_iou_thresh=cfg["fg_iou"], bg_iou_thresh=cfg["bg_iou"],
        score_thresh=0.001, detections_per_img=500, topk_candidates=1000)
    generator = anchor_generator(cfg["anchor_base"], cfg["aspect_ratios"])
    anchors = generator.num_anchors_per_location()[0]
    head = model.head
    # Keep the pretrained conv towers; replace only the final prediction layers (classes, and boxes if the
    # number of anchors per location changed).
    head.classification_head.cls_logits = nn.Conv2d(256, anchors * NUM_CLASSES, 3, padding=1)
    head.classification_head.num_classes = NUM_CLASSES
    nn.init.normal_(head.classification_head.cls_logits.weight, std=0.01)
    nn.init.constant_(head.classification_head.cls_logits.bias, -4.595)   # prior probability 0.01
    if anchors != model.anchor_generator.num_anchors_per_location()[0]:
        head.regression_head.bbox_reg = nn.Conv2d(256, anchors * 4, 3, padding=1)
        nn.init.normal_(head.regression_head.bbox_reg.weight, std=0.01)
        nn.init.zeros_(head.regression_head.bbox_reg.bias)
    model.anchor_generator = generator
    head.regression_head._loss_type = cfg["box_loss"]
    set_focal_loss(cfg["focal_alpha"], cfg["focal_gamma"])
    return model


def build_legacy_torchvision(family: str, state: dict) -> nn.Module:
    """Rebuild a model saved by the Colab notebook (a bare state_dict) and load it.

    Faster R-CNN used TorchVision's defaults with a 9-way box predictor. RetinaNet had its whole classification
    head replaced by a fresh, unnormalised one (the flaw that build_torchvision no longer has), so that exact
    head is recreated here. Inference limits match the other models: score 0.001, up to 500 detections.
    """
    if family == "fasterrcnn":
        model = build_torchvision("fasterrcnn", {}, pretrained=False)
    else:
        model = retinanet_resnet50_fpn_v2(weights=None, weights_backbone=None, score_thresh=0.001,
                                          detections_per_img=500, topk_candidates=1000)
        anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(256, anchors, NUM_CLASSES)
    model.load_state_dict(state)
    return model


def save_checkpoint(path: Path, family: str, cfg: dict, model: nn.Module, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"family": family, "cfg": cfg, "model": model.state_dict(), **extra}, temporary)
    temporary.replace(path)   # never leave a half-written checkpoint behind


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def torchvision_detector(model: nn.Module, family: str, device: str):
    """detect(PIL image) -> [(class 0-7, score, x1, y1, x2, y2)] in original pixels."""
    shift = 1 if family == "fasterrcnn" else 0   # Faster R-CNN labels are 1-8, RetinaNet's are 0-7

    def detect(image):
        tensor = pil_to_tensor(image).float().div_(255).to(device)
        with torch.inference_mode():
            result = model([tensor])[0]
        return [(label - shift, score, *box) for label, score, box in
                zip(result["labels"].tolist(), result["scores"].tolist(), result["boxes"].tolist())]

    return detect


def make_detector(family: str, checkpoint: Path, device: str, imgsz: int = 640, conf: float = 0.001, max_det: int = 500):
    """Load a trained checkpoint (TorchVision .pt from 03, Ultralytics .pt from 04) behind detect(image)."""
    if family in TORCHVISION_FAMILIES:
        saved = load_checkpoint(checkpoint)
        if "family" in saved:                                   # written by 03_train_torchvision.py
            assert saved["family"] == family, f"checkpoint holds {saved['family']}, not {family}"
            model = build_torchvision(family, saved["cfg"], pretrained=False)
            model.load_state_dict(saved["model"])
        else:                                                   # written by the Colab notebook
            model = build_legacy_torchvision(family, saved["model_state_dict"])
        return torchvision_detector(model.to(device).eval(), family, device)
    from ultralytics import RTDETR, YOLO

    model = (RTDETR if family == "rtdetr" else YOLO)(str(checkpoint))
    target = 0 if device.startswith("cuda") else device

    def detect(image):
        boxes = model.predict(source=image, imgsz=imgsz, conf=conf, max_det=max_det, device=target, verbose=False)[0].boxes
        return [(int(c), float(s), *b) for c, s, b in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())]

    return detect
