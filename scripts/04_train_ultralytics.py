"""Stage 04: train an Ultralytics detector: YOLO (anchor-free) or RT-DETR (query-based).

Run stage 02 first. Extra Ultralytics settings go through --set key=value (any value Python can parse),
which is how loss weights, augmentation and learning rates are tuned one at a time:

  python scripts/04_train_ultralytics.py yolo26n.pt --name yolo26n-base --epochs 30 --hours 0.75
  python scripts/04_train_ultralytics.py yolo26n.pt --name yolo26n-hires --imgsz 960 --set box=10
  python scripts/04_train_ultralytics.py rtdetr-l.pt --name rtdetr-l-base --batch 4 --patience 3

Ultralytics validates on converted labels (no ignored-region handling), so its mAP is only for monitoring;
final numbers come from stage 05. results.csv per epoch is the mAP-over-training curve.
"""

import argparse
import ast
import shutil
from pathlib import Path

from common import REPO, RESULTS, RUNS, pick_device, save_json


def literal(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="weights or yaml, e.g. yolo26n.pt or rtdetr-l.pt")
    parser.add_argument("--name", required=True)
    parser.add_argument("--data", type=Path, default=REPO / "data" / "yolo" / "stride5" / "data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100, help="upper bound; --hours or --patience usually stops first")
    parser.add_argument("--hours", type=float, default=0, help="wall-clock budget in hours (rescales the LR schedule); 0 = off")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="extra Ultralytics train arguments")
    args = parser.parse_args()

    from ultralytics import RTDETR, YOLO

    device = pick_device(args.device)
    overrides = {k: literal(v) for k, v in (item.split("=", 1) for item in args.set)}
    options = dict(data=str(args.data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, patience=args.patience,
                   workers=args.workers, seed=args.seed, device=0 if device.startswith("cuda") else device,
                   project=str(RUNS), name=args.name, exist_ok=True, plots=False, verbose=False, **overrides)
    if args.hours:
        options["time"] = args.hours
    result_dir = RESULTS / "runs" / args.name
    save_json(result_dir / "config.json", {"model": args.model, "train_arguments": {k: str(v) for k, v in options.items()}})
    model = (RTDETR if "rtdetr" in args.model.lower() else YOLO)(args.model)
    model.train(**options)
    run_dir = RUNS / args.name
    if (run_dir / "results.csv").is_file():
        shutil.copy(run_dir / "results.csv", result_dir / "results.csv")
    print("best checkpoint:", run_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()
