"""Fast unit tests for the pure helpers (no network, no GPU): python -m unittest discover -s tests"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import index_annotations, load_coco, sample_frames, size_bucket  # noqa: E402
from data import DetectionDataset, export_yolo, mask_regions  # noqa: E402
from detectors import build_torchvision  # noqa: E402
from evaluation import coco_evaluate, load_gt, prediction_string, validate_submission, write_submission  # noqa: E402
from synthetic import make_dataset  # noqa: E402


class HelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = make_dataset(Path(cls.folder.name))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_frame_sampling_keeps_every_clip(self):
        images = load_coco(self.root, "train")["images"]
        sampled = sample_frames(images, 5)
        self.assertEqual([i["frame_id"] for i in sampled], [1, 6, 1, 6])
        self.assertEqual({i["video"] for i in sampled}, {"clip_a", "clip_b"})

    def test_size_buckets_follow_coco(self):
        self.assertEqual([size_bucket(31, 31), size_bucket(33, 33), size_bucket(100, 100)], ["small", "medium", "large"])

    def test_dataset_labels_and_flip(self):
        plain = DetectionDataset(self.root, "train", label_offset=-1)
        image, target = plain[0]
        self.assertEqual(target["labels"].min().item() >= 0 and target["labels"].max().item() < 8, True)
        self.assertEqual(len(target["labels"]), 5)   # the ignored region is not a training object
        flipped = DetectionDataset(self.root, "train", flip=True)
        torch.manual_seed(0)
        import random
        random.seed(1)
        for _ in range(10):
            _, t = flipped[0]
            self.assertTrue((t["boxes"][:, 2] > t["boxes"][:, 0]).all())

    def test_masking_paints_ignored_and_keeps_labelled(self):
        image = Image.new("RGB", (100, 100), (10, 20, 30))
        region = [{"bbox": [0, 0, 50, 50]}]
        keep = [{"bbox": [40, 40, 10, 10]}]
        out = np.asarray(mask_regions(image, region, keep))
        self.assertEqual(tuple(out[5, 5]), (114, 114, 114))
        self.assertEqual(tuple(out[45, 45]), (10, 20, 30))     # labelled box restored
        self.assertEqual(tuple(out[80, 80]), (10, 20, 30))     # outside everything

    def test_yolo_export(self):
        out = self.root / "yolo"
        stats = export_yolo(self.root, out, stride=5, val_stride=1, mask_ignored=True)
        self.assertEqual((stats["train"]["frames"], stats["val"]["frames"]), (4, 6))
        label = next((out / "labels" / "train").rglob("*.txt")).read_text().split()
        self.assertEqual(len(label) % 5, 0)
        self.assertTrue((out / "data.yaml").is_file())
        train_image = next((out / "images" / "train").rglob("*.jpg"))
        val_image = next((out / "images" / "val").rglob("*.jpg"))
        self.assertFalse(train_image.is_symlink())   # masked copy
        self.assertTrue(val_image.is_symlink())      # validation images stay untouched

    def test_perfect_predictions_score_one(self):
        coco = load_coco(self.root, "val")
        objects, _ = index_annotations(coco)
        detections = [{"image_id": i, "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
                      for i, annotations in objects.items() for a in annotations]
        metrics = coco_evaluate(load_gt(self.root / "val" / "annotations_coco.json"), detections, [i["id"] for i in coco["images"]])
        self.assertAlmostEqual(metrics["AP"], 1.0, places=3)
        self.assertEqual(coco_evaluate(load_gt(self.root / "val" / "annotations_coco.json"), [], [1])["AP"], 0.0)

    def test_metrics_match_pycocotools_at_100_detections(self):
        import contextlib
        import io
        import random
        from pycocotools.cocoeval import COCOeval
        coco = load_coco(self.root, "val")
        objects, _ = index_annotations(coco)
        rng = random.Random(0)
        detections = []
        for image_id, annotations in objects.items():
            for a in annotations:
                x, y, w, h = a["bbox"]
                detections.append({"image_id": image_id, "category_id": rng.randrange(1, 9) if rng.random() < 0.2 else a["category_id"],
                                   "bbox": [x + rng.uniform(-6, 6), y + rng.uniform(-6, 6), w + rng.uniform(-6, 6), h + rng.uniform(-6, 6)],
                                   "score": rng.random()})
        gt = load_gt(self.root / "val" / "annotations_coco.json")
        ids = [i["id"] for i in coco["images"]]
        ours = coco_evaluate(gt, detections, ids, max_dets=100)
        with contextlib.redirect_stdout(io.StringIO()):
            reference = COCOeval(gt, gt.loadRes(detections), "bbox")
            reference.params.imgIds = ids
            reference.evaluate(); reference.accumulate(); reference.summarize()
        self.assertAlmostEqual(ours["AP"], reference.stats[0], places=6)
        self.assertAlmostEqual(ours["AP50"], reference.stats[1], places=6)
        self.assertAlmostEqual(ours["AP_small"], reference.stats[3], places=6)
        self.assertLess(ours["AP"], 1.0)

    def test_prediction_string_clips_and_limits(self):
        text = prediction_string([(2, 0.9, -5, 10, 30, 50), (1, 0.5, 10, 10, 10.3, 10.2), (3, 0.7, 300, 100, 400, 150)], 320, 180)
        groups = text.split()
        self.assertEqual(len(groups), 12)                      # the sub-pixel box is dropped
        self.assertEqual(groups[:6], ["2", "0.90000", "0.0", "10.0", "30.0", "40.0"])
        self.assertEqual(groups[8:10], ["300.0", "100.0"])
        self.assertEqual(float(groups[10]), 20.0)              # clipped to the 320 px frame
        many = [(0, i / 1000, 0, 0, 5, 5) for i in range(600)]
        self.assertEqual(len(prediction_string(many, 100, 100).split()) // 6, 500)

    def test_submission_validation(self):
        test_csv = self.root / "test_images.csv"
        import csv
        ids = [(r["image_id"], r["width"]) for r in csv.DictReader(test_csv.open())]
        good = [(i, "0 0.50000 10.0 10.0 20.0 20.0 3 0.25000 5.5 5.5 4.0 4.0") for i, _ in ids]
        path = self.root / "good.csv"
        write_submission(good, path)
        self.assertEqual(validate_submission(path, test_csv)[0], [])
        write_submission([(i, "") for i, _ in ids], path)                 # empty rows are allowed
        self.assertEqual(validate_submission(path, test_csv)[0], [])
        for bad_text in ("9 0.5 1 1 5 5", "0 0.5 1 1 0 5", "0 1.5 1 1 5 5", "0 0.5 1 1 5", "0 0.5 300 1 50 5"):
            write_submission([(i, bad_text) for i, _ in ids], path)
            self.assertTrue(validate_submission(path, test_csv)[0], bad_text)
        write_submission(good[:-1], path)
        self.assertTrue(any("rows" in p for p in validate_submission(path, test_csv)[0]))
        write_submission(list(reversed(good)), path)
        self.assertTrue(validate_submission(path, test_csv)[0])

    def test_model_builders_expose_the_knobs(self):
        frcnn = build_torchvision("fasterrcnn", {"box_batch_size_per_image": 128, "rpn_post_nms_top_n_train": 500}, pretrained=False)
        self.assertEqual(frcnn.roi_heads.fg_bg_sampler.batch_size_per_image, 128)
        self.assertEqual(frcnn.rpn._post_nms_top_n["training"], 500)
        self.assertEqual(frcnn.roi_heads.box_predictor.cls_score.out_features, 9)
        retina = build_torchvision("retinanet", {"anchor_base": [16, 32, 64, 128, 256], "aspect_ratios": [0.5, 1, 2, 4], "box_loss": "giou"}, pretrained=False)
        self.assertEqual(retina.anchor_generator.num_anchors_per_location()[0], 12)
        self.assertEqual(retina.head.classification_head.cls_logits.out_channels, 12 * 8)
        self.assertEqual(retina.head.regression_head.bbox_reg.out_channels, 12 * 4)
        self.assertEqual(retina.head.regression_head._loss_type, "giou")
        retina.eval()
        with torch.inference_mode():
            self.assertEqual(len(retina([torch.rand(3, 180, 320)])), 1)   # forward pass works with the changed head


class EnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib

        cls.fuse = staticmethod(importlib.import_module("09_ensemble").fuse)

    def test_agreeing_models_merge_into_one_box(self):
        a = [(0, 0.9, 10, 10, 50, 50)]
        b = [(0, 0.7, 12, 10, 52, 50)]
        fused = self.fuse([a, b])
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0][1], 0.8)             # mean score over the two models
        self.assertTrue(10 < fused[0][2] < 12)               # coordinates lie between the inputs

    def test_box_found_by_one_model_is_down_weighted_and_classes_stay_apart(self):
        fused = self.fuse([[(0, 0.8, 0, 0, 20, 20)], [(1, 0.8, 0, 0, 20, 20)]])
        self.assertEqual(sorted(f[0] for f in fused), [0, 1])
        self.assertTrue(all(abs(f[1] - 0.4) < 1e-9 for f in fused))


if __name__ == "__main__":
    unittest.main()
