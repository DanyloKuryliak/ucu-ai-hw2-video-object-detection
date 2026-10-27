"""Stage 02: build the Ultralytics (YOLO / RT-DETR) dataset from the competition COCO files.

TorchVision models (stage 03) read the COCO files directly and need no preparation; they take the same
--stride and --mask-ignored options. Frames are sampled per clip: with --stride 5 the train set is every
5th frame (4,868 of 24,201), the sample used for all four models so far.
"""

import argparse
from pathlib import Path

from common import REPO, RESULTS, data_root, save_json
from data import export_yolo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "yolo" / "stride5", help="output folder (git-ignored)")
    parser.add_argument("--stride", type=int, default=5, help="keep every Nth training frame of each clip")
    parser.add_argument("--val-stride", type=int, default=1,
                        help="validation frames Ultralytics scores each epoch (1 = all 2,846; larger is faster)")
    parser.add_argument("--mask-ignored", action="store_true", help="paint ignored regions grey in the train images")
    args = parser.parse_args()
    stats = export_yolo(data_root(args.data), args.out, args.stride, args.val_stride, args.mask_ignored)
    save_json(RESULTS / "prepared" / f"{args.out.name}.json", {"args": {k: str(v) for k, v in vars(args).items()}, **stats})
    print(stats)
    print("data yaml:", args.out / "data.yaml")


if __name__ == "__main__":
    main()
