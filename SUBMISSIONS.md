# Kaggle submissions

Log every upload here. The two final submissions are chosen by validation score on all 2,846 validation frames, not only by the public leaderboard. Every file below passes `python scripts/06_predict_submit.py --validate-only submissions/<file>.csv` (3,811 rows, at most 500 detections per frame).

| Date | File | Model | Validation AP | Public LB | Notes |
|---|---|---|---:|---:|---|
| 2026-10-26 | `ens-5models.csv` | weighted box fusion of RT-DETR-L (new), RT-DETR-L (Colab), Faster R-CNN, RetinaNet (GIoU), YOLO26n (960 px) | 0.255 | 0.26656 | final submission 1 (best on validation and on the public board) |
| 2026-10-26 | `ens-3models.csv` | the two RT-DETR-L models + Faster R-CNN | 0.249 | 0.26288 | final submission 2 |
| not uploaded | `rtdetr-l-e6.csv` | RT-DETR-L, 6 epochs | 0.223 | | best single model; not uploaded, kept as a fallback |

Submit with `kaggle competitions submit -c multiple-object-video-detection-hw-2-ucu-ai-2026 -f submissions/<file>.csv -m "<message>"`, then fill the Public LB column. The public score is higher than the validation AP for both files, and the order of the two files is the same on both. The submission files are not tracked by git (`submissions/*.csv`).
