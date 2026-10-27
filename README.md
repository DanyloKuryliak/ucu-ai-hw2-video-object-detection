# UCU AI HW2 — Multiple Object Video Detection

Homework for the UCU AI 2026 course: detect eight object classes (person, bicycle, car, van, truck, tricycle, bus, motor) in drone video frames from VisDrone2019-VID. Kaggle competition: [multiple-object-video-detection-hw-2-ucu-ai-2026](https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026).

## Status

All four detector families were trained, tuned (two or more single changes each, including a loss change) and scored on the full validation split (2,846 frames, 7 clips) under one protocol; their detections are fused by an ensemble. Two ensemble files were uploaded to Kaggle: public score 0.26656 (five models) and 0.26288 (three models), see `SUBMISSIONS.md`. Details of every run are in `EXPERIMENTS.md`.

Best run of each family and the ensemble (validation, same protocol):

| Model | Family | AP | AP50 | AP small | ms per frame |
|---|---|---:|---:|---:|---:|
| Faster R-CNN (`frcnn-base`) | two-stage, anchor-based | 0.170 | 0.362 | 0.066 | 131 |
| RetinaNet, GIoU box loss (`retina-giou`) | one-stage, anchor-based | 0.158 | 0.313 | 0.055 | 111 |
| YOLO26n, 960 px (`yolo26n-hires960`) | one-stage, anchor-free | 0.162 | 0.308 | 0.062 | 12 |
| RT-DETR-L (`rtdetr-l-e6`) | query-based | 0.223 | 0.429 | 0.106 | 38 |
| **Ensemble of five models** (`ens-5models`) | weighted box fusion | **0.255** | 0.484 | 0.106 | 330 |

What changed the score: input size 960 for YOLO26n (+0.035), the GIoU box loss for RetinaNet (+0.007), and fusing models (+0.032 over the best single one). Fewer region proposals and a smaller RoI batch hurt Faster R-CNN; a larger box-loss weight hurt YOLO26n; masking ignored regions did not help RT-DETR-L.

## Pipeline

Run from the repository root. Every stage prints its options with `--help`.

| Stage | Script | What it does |
|---|---|---|
| 00 | `scripts/00_download_data.py` | downloads the 11 GB archive, unpacks only the frames the pipeline reads (about 5 GB), deletes the archive |
| 01 | `scripts/01_eda.py` | class balance, object sizes, density, ignored regions, sampling stride → `results/eda.json`, `images/eda_*` |
| 02 | `scripts/02_prepare_data.py` | Ultralytics dataset from the COCO files; `--mask-ignored` paints ignored regions grey in train images |
| 03 | `scripts/03_train_torchvision.py` | Faster R-CNN / RetinaNet with tuning options (RPN proposals, RoI sampling, anchors, IoU thresholds, focal loss, box loss, schedule) |
| 04 | `scripts/04_train_ultralytics.py` | YOLO26 / RT-DETR; any Ultralytics setting through `--set key=value` |
| 05 | `scripts/05_evaluate.py` | one scoring protocol for every model: AP, AP50, AP small/medium/large, per class, per clip, ms per frame |
| 06 | `scripts/06_predict_submit.py` | test predictions → `submissions/<name>.csv`, format check, prints the Kaggle submit command (never uploads) |
| 07 | `scripts/07_failure_cases.py` | recall by size and class, class confusions, worst frames drawn with misses and false positives |
| 08 | `scripts/08_figures.py` | figures in `images/` from saved results and real frames: data, tuning study, model comparison, training curves, predictions, failures |
| 09 | `scripts/09_ensemble.py` | weighted box fusion of several models, on validation predictions or on submission files |

Shared code: `common.py` (paths, sampling), `data.py` (datasets, masking, YOLO export), `detectors.py` (model builders, checkpoints), `evaluation.py` (COCO metrics, submission checks). Data and checkpoints stay out of git: `data/`, `runs/`, `submissions/*.csv`.

```bash
pip install torch torchvision ultralytics pycocotools matplotlib pillow pyyaml   # Python 3.11 or 3.12
python scripts/00_download_data.py          # needs the Kaggle token; about 5 GB left on disk
python scripts/01_eda.py --preview
python scripts/02_prepare_data.py
python scripts/04_train_ultralytics.py rtdetr-l.pt --name rtdetr-l-base --batch 4 --hours 1
python scripts/05_evaluate.py rtdetr runs/rtdetr-l-base/weights/best.pt --name rtdetr-l-base
python scripts/06_predict_submit.py rtdetr runs/rtdetr-l-base/weights/best.pt --name rtdetr-l-base
```

Tests: `python -m unittest discover -s tests` (seconds, no data needed). Set `HW2_SMOKE_ULTRALYTICS=1` to include the Ultralytics smoke test.

### Reproducing the study

The runs in `EXPERIMENTS.md` used an Apple GPU (`--device mps`, about 15 minutes per run); on a CUDA machine drop `--device` and add `--amp` for the TorchVision models.

```bash
python scripts/02_prepare_data.py --val-stride 14
python scripts/02_prepare_data.py --mask-ignored --val-stride 14 --out data/yolo/masked
python scripts/03_train_torchvision.py fasterrcnn --name frcnn-base --max-minutes 15 --flip --device mps
python scripts/03_train_torchvision.py retinanet --name retina-giou --max-minutes 15 --flip --device mps --box-loss giou
python scripts/04_train_ultralytics.py yolo26n.pt --name yolo26n-hires960 --imgsz 960 --hours 0.25 --batch 8 --device mps
python scripts/04_train_ultralytics.py rtdetr-l.pt --name rtdetr-l-e6 --batch 4 --epochs 6 --device mps --set warmup_epochs=1
python scripts/05_evaluate.py rtdetr runs/rtdetr-l-e6/weights/best.pt --name rtdetr-l-e6 --per-clip --save-predictions
python scripts/09_ensemble.py val runs/eval/<a>_predictions.json runs/eval/<b>_predictions.json --name ens
python scripts/09_ensemble.py test submissions/<a>.csv submissions/<b>.csv --name ens
```

## Layout

| Path | Contents |
|---|---|
| `notebooks/` | `01`-`06`: one per rubric item, executed, tables filled from `results/`; plus the original Colab log (parts 0-7) |
| `scripts/`, `tests/` | pipeline stages and tests |
| `results/` | `eda.json`, `model_metrics.json`, `notebook/` (numbers saved from the Colab runs), and per-run configs, histories and evaluations |
| `EXPERIMENTS.md`, `SUBMISSIONS.md` | experiment plan and log; Kaggle submission log |
| `report.md`, `images/` | the written report and its figures (`fig_*.png` from stage 08), published with GitHub Pages |

## Report site

The report is built from `report.md` by GitHub Actions. One-time setup: **Settings → Pages → Source: GitHub Actions**, then re-run the failed **Deploy report** job. `index.html`, `assets/` and `.github/` come from the course template and are unchanged.
