"""Stage 00: download the competition archive and unpack only what the pipeline reads.

The full competition is 11 GB. Training reads every Nth frame of each training clip (default 5), evaluation
reads all validation frames, and prediction reads all test frames, so only those images are unpacked, about
5 GB. The COCO annotation files and test_images.csv are always unpacked; YOLO label files are not needed
because stage 02 builds them from the COCO files. The archive is deleted afterwards unless --keep-zip is given.

  python scripts/00_download_data.py                  # train stride 5, all val, all test
  python scripts/00_download_data.py --train-stride 1 # every training frame (about 10 GB)
"""

import argparse
import json
import subprocess
import zipfile
from pathlib import Path

from common import REPO, SLUG, sample_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "data")
    parser.add_argument("--train-stride", type=int, default=5, help="unpack every Nth training frame of each clip")
    parser.add_argument("--zip", type=Path, help="use this archive instead of downloading")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    archive = args.zip or args.out / "_archive" / f"{SLUG}.zip"
    if not archive.is_file():
        archive.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "competitions", "download", "-c", SLUG, "-p", str(archive.parent), "-q"], check=True)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        small = [n for n in names if n.endswith((".json", ".csv")) and "/labels/" not in n and "/ignore/" not in n]
        z.extractall(args.out, members=small)
        train = json.loads((args.out / "train" / "annotations_coco.json").read_text())["images"]
        keep_train = {f"train/{i['file_name']}" for i in sample_frames(train, args.train_stride)}
        images = [n for n in names if n.endswith(".jpg") and (
            n.startswith(("val/images/", "test/images/")) or n in keep_train)]
        done = 0
        for name in images:
            target = args.out / name
            if not target.is_file() or target.stat().st_size != z.getinfo(name).file_size:
                z.extract(name, args.out)
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(images)} images", flush=True)
    counts = {s: sum(1 for _ in (args.out / s / "images").rglob("*.jpg")) for s in ("train", "val", "test")}
    expected = {"train": len(keep_train), "val": len(json.loads((args.out / "val" / "annotations_coco.json").read_text())["images"]),
                "test": sum(1 for _ in open(args.out / "test_images.csv")) - 1}
    print("images on disk:", counts, "| expected:", expected)
    assert all(counts[s] >= expected[s] for s in counts), "some images are missing from the archive"
    if not args.keep_zip and not args.zip:
        archive.unlink()
        print("deleted", archive)


if __name__ == "__main__":
    main()
