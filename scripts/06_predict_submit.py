"""Stage 06: write and validate a Kaggle submission CSV from a checkpoint (or validate an existing CSV).

This script never uploads anything; it prints the submit command for you to run once the file is valid.

  python scripts/06_predict_submit.py rtdetr runs/rtdetr-l-base/weights/best.pt --name rtdetr-l-base
  python scripts/06_predict_submit.py --validate-only submissions/some-file.csv
"""

import argparse
import csv
from pathlib import Path

from PIL import Image

from common import REPO, SLUG, data_root, pick_device
from detectors import FAMILIES, make_detector
from evaluation import MAX_DETS, prediction_string, validate_submission, write_submission


def check(path: Path, test_csv: Path) -> bool:
    problems, stats = validate_submission(path, test_csv)
    print(f"{path}: {stats}")
    for problem in problems:
        print("  PROBLEM:", problem)
    print("submission is valid" if not problems else f"{len(problems)} problem(s) found; do not submit")
    return not problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", nargs="?", choices=FAMILIES)
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("--name", help="writes submissions/<name>.csv")
    parser.add_argument("--validate-only", type=Path, metavar="CSV")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=MAX_DETS)
    parser.add_argument("--device")
    args = parser.parse_args()

    root = data_root(args.data)
    test_csv = root / "test_images.csv"
    if args.validate_only:
        raise SystemExit(0 if check(args.validate_only, test_csv) else 1)
    assert args.family and args.checkpoint and args.name, "family, checkpoint and --name are required"
    device = pick_device(args.device)
    detect = make_detector(args.family, args.checkpoint, device, args.imgsz, args.conf, args.max_det)
    with test_csv.open(newline="") as f:
        test = list(csv.DictReader(f))
    rows = []
    for index, row in enumerate(test, 1):
        with Image.open(root / row["file_name"]) as source:
            frame = source.convert("RGB")
        if frame.size != (int(row["width"]), int(row["height"])):
            print(f"  note: {row['image_id']} is {frame.size}, test_images.csv says {row['width']}x{row['height']}")
        rows.append((row["image_id"], prediction_string(detect(frame), *frame.size, args.max_det)))
        if index % 500 == 0:
            print(f"  {index}/{len(test)} frames", flush=True)
    output = REPO / "submissions" / f"{args.name}.csv"
    write_submission(rows, output)
    if check(output, test_csv):
        print(f'submit with:\n  kaggle competitions submit -c {SLUG} -f {output} -m "{args.name}"')
        print("then add a row to SUBMISSIONS.md")


if __name__ == "__main__":
    main()
