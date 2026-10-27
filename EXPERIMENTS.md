# Experiments

Working log of every run. Numbers come from `results/eval/<run>.json` (written by `scripts/05_evaluate.py`); per-run configs and training curves are in `results/runs/<run>/`.

## Protocol (same for every run)

- **Train:** every 5th frame of each of the 56 training clips (4,868 frames), seed 42, COCO-pretrained weights.
- **Budget:** 15 minutes of training per CNN run and per YOLO run, on one machine (a Mac Mini, Apple GPU, `--device mps`). RT-DETR-L is much slower per step: its baseline ran 6 epochs (92 minutes, on a MacBook) and the masked run 45 minutes on the Mac Mini.
- **Score:** all 2,846 frames of the 7 validation clips, COCO AP with up to 500 detections per frame, ignored regions respected. Quick checks during training use a validation subset and only pick the checkpoint.
- **One change per run** against the baseline of the same family. Differences below about 0.005 AP are within run-to-run noise (single seed).
- **Speed** is ms per frame at batch size 1 on a MacBook GPU (the four best runs were scored on that one machine, so they compare with each other; absolute times are not T4 times).

## The earlier Colab runs (pilots, kept as the starting point)

| Run | AP | AP50 | AP small | Notes |
|---|---:|---:|---:|---|
| Faster R-CNN, step 3,755 | 0.140 | 0.314 | 0.061 | the Colab score was 0.124 under a stricter score threshold |
| RetinaNet, step 1,000 | 0.103 | 0.195 | 0.041 | the whole classification head was re-created, discarding the pretrained layers |
| YOLO26n, epoch 4 | 0.131 | 0.254 | 0.041 | ignored regions unmasked, 640 px |
| RT-DETR-L, epoch 2 | 0.225 | 0.426 | 0.086 | warm-up lasted until epoch 3, so the schedule never played out |

## Tuning study (this repository's scripts)

| Family | Run | Change | AP | AP50 | AP small | ms/frame | Train min | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| Two-stage | `frcnn-base` | TorchVision defaults | 0.170 | 0.362 | 0.066 | 131 | 15.6 | best of family |
| | `frcnn-rpn300` | RPN proposals kept 2000 → 300 | 0.137 | 0.289 | 0.061 | 109 | 15.6 | worse |
| | `frcnn-roi128` | RoI samples per image 512 → 128 | 0.151 | 0.328 | 0.063 | 110 | 15.6 | worse |
| One-stage, anchors | `retina-base` | pretrained head layers kept | 0.151 | 0.305 | 0.047 | 96 | 15.5 | |
| | `retina-anchors16` | anchor sizes 32-512 → 16-256 | 0.148 | 0.324 | 0.049 | 96 | 15.5 | no change in AP, better AP50 |
| | `retina-giou` | box loss L1 → GIoU | 0.158 | 0.313 | 0.055 | 111 | 15.5 | best of family (+0.007) |
| Anchor-free | `yolo26n-base` | 640 px | 0.128 | 0.243 | 0.040 | 13 | 15.1 | |
| | `yolo26n-hires960` | input 640 → 960 px | 0.162 | 0.308 | 0.062 | 12 | 15.1 | best of family (+0.035) |
| | `yolo26n-box10` | box-loss weight 7.5 → 10 | 0.108 | 0.218 | 0.038 | 13 | 15.1 | worse |
| Query-based | `rtdetr-l-e6` | 6 epochs, 1-epoch warm-up | 0.223 | 0.429 | 0.106 | 38 | 91.7 | best single model |
| | `rtdetr-l-masked` | ignored regions painted grey in training images | 0.217 | 0.413 | 0.098 | 44 | 45.3 | slightly worse, but also a shorter run |
| | `rtdetr-l-e6-infer800` | same weights, input 800 px at test time | 0.221 | 0.429 | 0.105 | 52 | - | no gain |
| | `rtdetr-l-e6-infer960` | same weights, input 960 px at test time | 0.223 | 0.434 | 0.116 | 67 | - | better small objects, worse large |

Run-by-run commands are in `runs/logs` (not tracked); each one is `03_train_torchvision.py <family> --name <run> --max-minutes 15 <flag>` or `04_train_ultralytics.py <weights> --name <run> --hours 0.25 [--imgsz ...] [--set key=value]`, followed by `05_evaluate.py ... --per-clip`.

Failure-case analysis (`07_failure_cases.py`) was run on the best run of every family; results in `results/failures/`, overlays described in notebooks 02-05.

## Own pipeline: weighted box fusion (`scripts/09_ensemble.py`)

Detections of several models are fused per frame and class: boxes with IoU above 0.55 are merged, coordinates are the score-weighted mean, and the score is the mean over models, so a box that only one model finds is down-weighted.

| Ensemble | AP | AP50 | AP small | AP large | ms/frame (sum) |
|---|---:|---:|---:|---:|---:|
| `ens-rtdetr2`: `rtdetr-l-e6` + the Colab RT-DETR-L | 0.241 | 0.466 | 0.105 | 0.424 | 76 |
| `ens-3models`: + `frcnn-base` | 0.249 | 0.480 | 0.106 | 0.443 | 207 |
| `ens-5models`: + `retina-giou` + `yolo26n-hires960` | **0.255** | 0.484 | 0.106 | 0.451 | 330 |

Other pipeline ideas tried: masking ignored regions (no gain), higher inference size for RT-DETR (small objects up, large down, AP flat), input 960 for YOLO (the biggest single gain for a CNN).
