"""End-to-end smoke test: every stage script runs on tiny synthetic data (CPU, a few seconds per model).

The TorchVision path always runs. The Ultralytics path needs to download yolo26n.pt, so it only runs when
HW2_SMOKE_ULTRALYTICS=1:   HW2_SMOKE_ULTRALYTICS=1 python -m unittest tests.test_smoke_pipeline
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic import make_dataset  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RESULTS = Path(__file__).resolve().parents[1] / "results"


def run(script: str, *args: str):
    result = subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(f"{script} failed:\n{result.stdout[-1500:]}\n{result.stderr[-2500:]}")
    return result.stdout


def cleanup(tag: str):
    """Remove everything the smoke runs wrote under results/, runs/ and submissions/."""
    import shutil
    base = SCRIPTS.parent
    for folder in (RESULTS / "runs", RESULTS / "eval", RESULTS / "failures", RESULTS / "prepared", base / "runs", base / "runs" / "eval",
                   base / "runs" / "failures", base / "submissions"):
        for path in folder.glob(f"{tag}*") if folder.is_dir() else []:
            shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)


class SmokePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = make_dataset(Path(cls.folder.name) / "data")
        cls.tag = f"smoke-{os.getpid()}"
        cls.created = []

    @classmethod
    def tearDownClass(cls):
        cleanup(cls.tag)
        cls.folder.cleanup()

    def pipeline(self, family: str, *train_extra: str):
        name = f"{self.tag}-{family}"
        data = ["--data", str(self.root)]
        run("03_train_torchvision.py", family, "--name", name, *data, "--no-pretrained", "--device", "cpu", "--stride", "1",
            "--batch-size", "2", "--workers", "0", "--max-steps", "4", "--eval-every", "2", "--warmup-steps", "1", "--val-stride", "2",
            "--min-size", "180", "--max-size", "320", "--mask-ignored", "--flip", *train_extra)
        best = SCRIPTS.parent / "runs" / name / "best.pt"
        self.assertTrue(best.is_file())
        history = json.loads((RESULTS / "runs" / name / "history.json").read_text())
        self.assertEqual([h["step"] for h in history], [2, 4])
        run("05_evaluate.py", family, str(best), "--name", name, *data, "--device", "cpu", "--per-clip", "--save-predictions")
        report = json.loads((RESULTS / "eval" / f"{name}.json").read_text())
        self.assertEqual(report["frames"], 6)
        self.assertIn("clip_c", report["per_clip"])
        run("06_predict_submit.py", family, str(best), "--name", name, *data, "--device", "cpu")
        csv_path = SCRIPTS.parent / "submissions" / f"{name}.csv"
        self.assertTrue(csv_path.is_file())
        run("06_predict_submit.py", "--validate-only", str(csv_path), *data)
        run("07_failure_cases.py", "--name", name, *data, "--score-thr", "0.05", "--top", "2")
        self.assertTrue((RESULTS / "failures" / f"{name}.json").is_file())
        self.created.append(name)

    def test_fasterrcnn_stages(self):
        self.pipeline("fasterrcnn", "--rpn-post-nms-top-n-train", "200", "--box-batch-size-per-image", "64")

    def test_retinanet_stages(self):
        self.pipeline("retinanet", "--box-loss", "giou", "--focal-gamma", "1.5", "--anchor-base", "8,16,32,64,128", "--optimizer", "adamw", "--lr", "0.0001")

    @unittest.skipUnless(os.environ.get("HW2_SMOKE_ULTRALYTICS") == "1", "set HW2_SMOKE_ULTRALYTICS=1 (downloads yolo26n.pt)")
    def test_ultralytics_stages(self):
        out = Path(self.folder.name) / "yolo"
        data = ["--data", str(self.root)]
        run("02_prepare_data.py", *data, "--out", str(out / self.tag), "--stride", "1", "--mask-ignored")
        name = f"{self.tag}-yolo"
        run("04_train_ultralytics.py", "yolo26n.pt", "--name", name, "--data", str(out / self.tag / "data.yaml"), "--imgsz", "160",
            "--epochs", "1", "--batch", "4", "--workers", "0", "--device", "cpu", "--set", "amp=False", "pretrained=True")
        best = SCRIPTS.parent / "runs" / name / "weights" / "best.pt"
        self.assertTrue(best.is_file())
        run("05_evaluate.py", "yolo", str(best), "--name", name, *data, "--device", "cpu", "--imgsz", "160")
        run("06_predict_submit.py", "yolo", str(best), "--name", name, *data, "--device", "cpu", "--imgsz", "160")
        self.created.append(name)


if __name__ == "__main__":
    unittest.main()
