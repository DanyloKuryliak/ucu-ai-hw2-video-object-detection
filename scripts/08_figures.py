"""Stage 08: figures for the write-up, drawn from saved results (no training or GPU needed).

Reads results/eda.json, results/model_metrics.json, results/notebook/*, and, for the picture grids, real
frames in data/ plus predictions from `05_evaluate.py --save-predictions`. Figures that need a missing input
are skipped with a note. Colours: one fixed colour per model and per split, taken from a validated palette;
every bar carries its value because the aqua and yellow slots have low contrast on white.

  python scripts/08_figures.py                       # everything available
  python scripts/08_figures.py --only data           # data-understanding figures only
  python scripts/08_figures.py --predictions runs/eval/<name>_predictions.json --failures runs/failures/<name>
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image

from common import CLASS_NAMES, FIGURES, REPO, RESULTS, data_root, index_annotations, load_coco

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"
MODEL_COLOUR = {"Faster R-CNN": "#2a78d6", "RetinaNet": "#eb6834", "YOLO26n": "#1baf7a", "RT-DETR-L": "#eda100", "Ensemble": "#52514e"}
MODEL_FAMILY = {"Faster R-CNN": "two-stage", "RetinaNet": "one-stage, anchors", "YOLO26n": "anchor-free", "RT-DETR-L": "query-based"}
SPLIT_COLOUR = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
MEASURE_COLOUR = ["#2a78d6", "#eb6834", "#1baf7a"]
BASE = "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#c9c8c3", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def finish(fig, name: str, title: str, subtitle: str = "", note: str = "", legend=None):
    """Reserve room (in inches) for title, subtitle, optional legend and source note, lay out, save to images/."""
    h = fig.get_figheight()
    top = 0.85 if subtitle else 0.5
    bottom = 0.42 + (0.4 if legend else 0)
    fig.tight_layout(rect=(0, bottom / h, 1, 1 - top / h))
    fig.text(0.01, 1 - 0.1 / h, title, fontsize=13, fontweight="bold", va="top", ha="left")
    if subtitle:
        fig.text(0.01, 1 - 0.45 / h, subtitle, fontsize=9.5, color=INK2, va="top", ha="left")
    if legend:
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.36 / h), ncol=len(legend), frameon=False)
    if note:
        fig.text(0.01, 0.1 / h, note, fontsize=8, color=INK2, va="bottom", ha="left")
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print("wrote", FIGURES / name)


def labels(ax, bars, fmt="{:.3f}", horizontal=False, pad=3):
    for bar in bars:
        value = bar.get_width() if horizontal else bar.get_height()
        if horizontal:
            ax.annotate(fmt.format(value), (value, bar.get_y() + bar.get_height() / 2), xytext=(pad, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK)
        else:
            ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value), xytext=(0, pad),
                        textcoords="offset points", ha="center", fontsize=8.5, color=INK)


# ---- results of the four models ----------------------------------------------------------------
def fig_progress():
    steps = json.loads((RESULTS / "notebook" / "milestones.json").read_text())["full_validation_in_order"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(steps))[::-1]
    bars = ax.barh(y, [s["mAP"] for s in steps], color=[MODEL_COLOUR[s["run"]] for s in steps], height=0.62)
    ax.set_yticks(y, [f"{s['run']}  ·  {s['stage']}" for s in steps])
    ax.set(xlim=(0, 0.27), xlabel="mAP on all 2,846 validation frames (higher is better)")
    ax.grid(axis="y", visible=False)
    labels(ax, bars, "{:.3f}", horizontal=True)
    finish(fig, "fig_progress_timeline.png", "How the scores moved, in the order the runs were made",
           "Pilot runs first, then the four detector families on the same 4,868-frame training sample",
           "Source: notebook outputs as recorded then (Faster R-CNN was scored with a higher score threshold; see fig_model_comparison). Budgets differ between rows.")


def fig_models():
    scored = eval_results()
    if scored:
        metrics = {name: {"mAP": r["AP"], "AP50": r["AP50"], "AP_small": r["AP_small"]} for name, r in scored.items()}
        names = {name: name for name in scored}
        source = "Source: results/eval/*.json, one protocol for every model (score threshold 0.001, up to 500 detections)."
    else:
        metrics = json.loads((RESULTS / "model_metrics.json").read_text())["models"]
        names = {"Faster R-CNN ResNet-50 FPN": "Faster R-CNN", "RetinaNet ResNet-50 FPN v2": "RetinaNet", "YOLO26n": "YOLO26n", "RT-DETR-L": "RT-DETR-L"}
        source = "Source: results/model_metrics.json (notebook runs)."
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=False)
    display = {"Faster R-CNN": "Faster\nR-CNN\ntwo-stage", "RetinaNet": "RetinaNet\none-stage,\nanchors", "YOLO26n": "YOLO26n\nanchor-free", "RT-DETR-L": "RT-DETR-L\nquery-based"}
    for ax, (key, title) in zip(axes, [("mAP", "mAP"), ("AP50", "AP50 (loose boxes)"), ("AP_small", "AP small objects")]):
        values = [metrics[full][key] for full in names]
        bars = ax.bar(range(4), values, color=[MODEL_COLOUR[n] for n in names.values()], width=0.62)
        ax.set_xticks(range(4), [display[n] for n in names.values()], fontsize=8)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(axis="x", visible=False)
        labels(ax, bars)
    finish(fig, "fig_model_comparison.png", "Four detector families on the same validation clips",
           "All 2,846 frames from 7 held-out clips, COCO metrics, higher is better",
           source + " Trained on every 5th frame of the training clips, short and unequal budgets.")


def fig_rtdetr_training():
    with (RESULTS / "notebook" / "rtdetr_l_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    epoch = [int(r["epoch"]) for r in rows]
    col = lambda key: [float(r[key]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    ax = axes[0]
    ax.plot(epoch, col("metrics/mAP50-95(B)"), color=MODEL_COLOUR["RT-DETR-L"], marker="o", lw=2, label="mAP50-95")
    ax.plot(epoch, col("metrics/mAP50(B)"), color=MODEL_COLOUR["Faster R-CNN"], marker="o", lw=2, label="mAP50")
    for e, v in zip(epoch, col("metrics/mAP50-95(B)")):
        ax.annotate(f"{v:.3f}", (e, v), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8.5)
    ax.set(xlabel="epoch", xticks=epoch, ylim=(0.1, 0.55))
    ax.set_title("Validation score per epoch", loc="left", fontsize=10.5)
    ax.legend(loc="upper left", ncol=2)
    axes[1].plot(epoch, col("lr/pg0"), color=INK2, marker="o", lw=2)
    axes[1].set(xlabel="epoch", xticks=epoch)
    axes[1].set_title("Learning rate per epoch", loc="left", fontsize=10.5)
    axes[1].axvspan(0.5, 3.0, color=GRID, alpha=0.6, lw=0)
    axes[1].annotate("still rising\n(warm-up)", (1.6, max(col("lr/pg0")) * 0.55), ha="center", fontsize=9, color=INK2)
    axes[2].plot(epoch, col("train/giou_loss"), color=MODEL_COLOUR["Faster R-CNN"], marker="o", lw=2, label="train")
    axes[2].plot(epoch, col("val/giou_loss"), color=MODEL_COLOUR["RetinaNet"], marker="o", lw=2, label="validation")
    axes[2].set(xlabel="epoch", xticks=epoch)
    axes[2].set_title("Box (GIoU) loss per epoch", loc="left", fontsize=10.5)
    axes[2].legend()
    finish(fig, "fig_rtdetr_training.png", "RT-DETR-L training curve",
           "The best epoch was 2, and the learning rate was still rising until epoch 3 of a 4-epoch run",
           "Source: results/notebook/rtdetr_l_results.csv (Ultralytics' own validation on converted labels, so slightly different from the COCO re-score).")


def fig_quick_validation():
    history = json.loads((RESULTS / "notebook" / "milestones.json").read_text())["quick_validation_219_frames"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name in ("Faster R-CNN", "RetinaNet"):
        steps = [p["step"] for p in history[name]]
        values = [p["mAP"] for p in history[name]]
        ax.plot(steps, values, color=MODEL_COLOUR[name], marker="o", lw=2, ms=7)
        for s, v in zip(steps, values):
            ax.annotate(f"{v:.3f}", (s, v), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5)
        ax.annotate(name, (steps[-1], values[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK)
    ax.set(xlabel="training step", ylabel="mAP, quick check", xlim=(500, 4300), ylim=(0.04, 0.15))
    finish(fig, "fig_quick_validation.png", "Quick validation during training",
           "Faster R-CNN stayed flat; RetinaNet fell after step 1,000 and its best checkpoint was kept",
           "Same 219 held-out frames for both models; a cheaper check than the full validation, so values differ from the other figures.")


def fig_per_class():
    rows = json.loads((RESULTS / "notebook" / "rtdetr_l_classwise.json").read_text())
    rows = sorted(rows, key=lambda r: r["mAP"])
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(rows))
    for offset, key, name, colour in zip((-0.26, 0, 0.26), ("mAP", "AP50", "AP_small"), ("mAP", "AP50", "AP small objects"), MEASURE_COLOUR):
        values = [r[key] if r[key] is not None else 0 for r in rows]
        bars = ax.barh(y + offset, values, 0.24, color=colour, label=name)
        for bar, row in zip(bars, rows):
            if row[key] is None:
                ax.annotate("n/a", (0, bar.get_y() + bar.get_height() / 2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
        labels(ax, [b for b, r in zip(bars, rows) if r[key] is not None], "{:.2f}", horizontal=True)
    ax.set_yticks(y, [f"{r['class']}  ({r['boxes']:,} val boxes)" for r in rows])
    ax.set(xlim=(0, 0.75), xlabel="average precision (higher is better)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    finish(fig, "fig_rtdetr_per_class.png", "RT-DETR-L results by class",
           "Sorted by mAP; bicycles are found but almost never when small, buses have no small examples in validation",
           "Source: results/notebook/rtdetr_l_classwise.json (full validation, 2,846 frames).")


# ---- the data -----------------------------------------------------------------------------------
def eda():
    return json.loads((RESULTS / "eda.json").read_text())


def fig_data_classes():
    s = eda()["summary"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(CLASS_NAMES))
    for offset, split in ((-0.2, "train"), (0.2, "val")):
        counts = [s[split]["class_counts"].get(n, 0) for n in CLASS_NAMES]
        bars = ax.barh(y + offset, counts, 0.36, color=SPLIT_COLOUR[split], label=f"{split} ({s[split]['objects']:,} objects)")
        labels(ax, bars, "{:,}", horizontal=True)
    ax.set_yticks(y, CLASS_NAMES)
    ax.invert_yaxis()
    ax.set(xlabel="labelled objects (thousands)", xlim=(0, 620000))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    finish(fig, "fig_data_classes.png", "Class balance: cars and people dominate",
           "Rare classes (bus, truck) have far fewer examples, and the validation mix differs from training",
           "Source: results/eda.json (ignored regions are not counted).")


def fig_data_sizes():
    s = eda()["summary"]["train"]["sqrt_area_px_by_class"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    stats = [{"label": n, "whislo": s[n]["p5"], "q1": s[n]["p25"], "med": s[n]["p50"], "q3": s[n]["p75"], "whishi": s[n]["p95"], "fliers": []}
             for n in CLASS_NAMES if n in s]
    ax.bxp(stats, positions=range(len(stats)), vert=False, widths=0.5, showfliers=False, patch_artist=True,
           boxprops={"facecolor": BASE, "alpha": 0.35, "edgecolor": BASE}, medianprops={"color": INK, "lw": 2},
           whiskerprops={"color": BASE}, capprops={"color": BASE})
    ax.set_yticks(range(len(stats)), [d["label"] for d in stats])
    ax.set_ylim(len(stats) - 0.5, -1.1)
    for edge, text in ((32, "small | medium"), (96, "medium | large")):
        ax.axvline(edge, color=INK2, ls="--", lw=1)
        ax.annotate(text, (edge, -0.8), xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=INK2, va="center")
    ax.set(xscale="log", xlabel="object size in original pixels, sqrt(width x height), log scale", xlim=(8, 400))
    ax.set_xticks([10, 20, 32, 50, 96, 200, 400], ["10", "20", "32", "50", "96", "200", "400"])
    ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    finish(fig, "fig_data_sizes.png", "Object size by class (training set)",
           "Box = middle half of objects, line = median, whiskers = 5th to 95th percentile",
           "Source: results/eda.json. Dashed lines are the COCO small/medium/large limits (32 and 96 px).")


def fig_data_density():
    s = eda()["summary"]
    clips = [(c["objects_per_frame"], name, split) for split in ("train", "val") for name, c in s[split]["per_clip"].items()]
    clips.sort()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(range(len(clips)), [c[0] for c in clips], color=[SPLIT_COLOUR[c[2]] for c in clips], width=0.75)
    ax.set(xlabel=f"clips, sorted by density ({len(clips)} clips)", ylabel="labelled objects per frame", xticks=[], xlim=(-1, len(clips)))
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=SPLIT_COLOUR["train"], label="train clips"), Patch(color=SPLIT_COLOUR["val"], label="validation clips")], loc="upper left")
    finish(fig, "fig_data_density.png", "Scene density varies a lot between clips",
           f"From about {clips[0][0]:.0f} to {clips[-1][0]:.0f} labelled objects per frame on average",
           "Source: results/eda.json.")


def fig_data_resolution():
    s = eda()["summary"]
    sizes, raw = {}, {}
    for split in ("train", "val", "test"):
        total = sum(s[split]["image_sizes"].values())
        for res, n in s[split]["image_sizes"].items():
            w, h = map(int, res.split("x"))
            key = (round(w / 8), round(h / 8))             # sizes that differ by a pixel are the same video size
            sizes.setdefault(key, {})
            sizes[key][split] = sizes[key].get(split, 0) + n / total
            raw.setdefault(key, {})[res] = raw.get(key, {}).get(res, 0) + n
    names = {k: max(v, key=v.get) for k, v in raw.items()}
    order = sorted(sizes, key=lambda k: k[0] * k[1])
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(order))
    for offset, split in ((-0.27, "train"), (0, "val"), (0.27, "test")):
        bars = ax.bar(x + offset, [sizes[r].get(split, 0) for r in order], 0.26, color=SPLIT_COLOUR[split], label=split)
        labels(ax, [b for b, r in zip(bars, order) if sizes[r].get(split)], "{:.0%}")
    ax.set_xticks(x, [names[r] for r in order], fontsize=9)
    ax.set(xlabel="image size in pixels (width x height), small to large", ylabel="share of the split's frames")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="x", visible=False)
    ax.legend()
    finish(fig, "fig_data_resolution.png", "Test images are smaller than training and validation images",
           "Test frames reach only 1920 px wide; validation has a 3840x2160 clip making up a third of its frames",
           "Source: results/eda.json. Sizes within a pixel of each other (e.g. 1360x764 and 1360x765) are merged.")


def fig_data_ignored():
    s = eda()["summary"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for split in ("train", "val"):
        pts = [(c["objects"] / c["frames"], c["ignored"] / c["frames"]) for c in s[split]["per_clip"].values()]
        ax.scatter(*zip(*pts), color=SPLIT_COLOUR[split], s=42, edgecolor=SURFACE, linewidth=1.2, label=f"{split} clips")
    share = {k: s[k]["ignored_regions"] / (s[k]["ignored_regions"] + s[k]["objects"]) for k in ("train", "val")}
    ax.set(xlabel="labelled objects per frame", ylabel="ignored regions per frame")
    ax.legend(loc="upper left")
    finish(fig, "fig_data_ignored.png", "Ignored regions are common and differ a lot between clips",
           f"{share['train']:.0%} of training annotations and {share['val']:.0%} of validation annotations are ignored regions (one dot per clip)",
           "Source: results/eda.json. Detections inside ignored regions are not penalised in scoring.")


def fig_data_stride():
    p = eda()["track_persistence"]["train"]
    strides = [int(k.split("_")[1]) for k in p]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title, fmt in ((axes[0], "object_still_present", "Objects still there k frames later", "{:.0%}"),
                                (axes[1], "median_shift_in_box_sizes", "How far a typical object moved (in box sizes)", "{:.2f}")):
        values = [p[f"stride_{k}"][key] for k in strides]
        ax.plot(strides, values, color=BASE, marker="o", lw=2, ms=7)
        for k, v in zip(strides, values):
            ax.annotate(fmt.format(v), (k, v), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5)
        ax.axvline(5, color=INK2, ls="--", lw=1)
        ax.set(xlabel="gap between frames k", xticks=strides)
        ax.set_title(title, loc="left", fontsize=10.5)
    axes[0].set(ylim=(0.85, 1.03), yticks=[0.85, 0.90, 0.95, 1.00])
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[1].set_ylim(0, 0.95)
    axes[0].annotate("every 5th frame\n(used)", (5, 0.87), xytext=(8, 0), textcoords="offset points", fontsize=8.5, color=INK2)
    finish(fig, "fig_data_stride.png", "Neighbouring frames are highly redundant",
           "Why sampling every Nth frame of each clip is a reasonable shortcut (the cost of it was not measured)",
           "Source: results/eda.json, training clips, objects followed by their track id.")


# ---- pictures from real frames ------------------------------------------------------------------
def load_val(root):
    coco = load_coco(root, "val")
    objects, ignored = index_annotations(coco)
    return coco, objects, ignored


def draw(ax, image, boxes, colour, lw=1.2):
    ax.imshow(image)
    for x, y, w, h in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=colour, linewidth=lw))
    ax.axis("off")


def local_frames(root, coco, wanted=None):
    frames = [i for i in coco["images"] if (root / "val" / i["file_name"]).is_file()]
    return frames if wanted is None else [f for f in frames if f["id"] in wanted]


def thumb(path, width=1100):
    with Image.open(path) as source:
        image = source.convert("RGB")
    scale = width / image.width
    return image.resize((width, int(image.height * scale))), scale


def fig_qual_predictions(root, predictions_path):
    coco, objects, ignored = load_val(root)
    preds = json.loads(Path(predictions_path).read_text())
    by_image = {}
    for d in preds:
        if d["score"] >= 0.3:
            by_image.setdefault(d["image_id"], []).append(d)
    seen = {}
    for info in local_frames(root, coco, set(by_image)):
        seen.setdefault(info["video"], info)              # one frame per clip
    chosen = sorted(seen.values(), key=lambda i: abs(len(objects[i["id"]]) - 40))[:3]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(13, 3.9 * len(chosen)))
    for row, info in enumerate(chosen):
        image, k = thumb(root / "val" / info["file_name"])
        scale = lambda boxes: [[v * k for v in b] for b in boxes]
        draw(axes[row, 0], image, scale([a["bbox"] for a in objects[info["id"]]]), "#2fbf71")
        draw(axes[row, 1], image, scale([d["bbox"] for d in by_image[info["id"]]]), "#2a78d6")
        axes[row, 0].set_title(f"{info['video']}, {info['width']}x{info['height']}  ·  ground truth ({len(objects[info['id']])} objects)", loc="left", fontsize=9.5)
        axes[row, 1].set_title(f"RT-DETR-L predictions, score >= 0.3 ({len(by_image[info['id']])} boxes)", loc="left", fontsize=9.5)
    finish(fig, "fig_qual_predictions.png", "Ground truth next to RT-DETR-L predictions",
           "Real validation frames from three different clips", "Source: the notebook's RT-DETR-L checkpoint on frames from data/val; one frame per clip, chosen for a moderate object count.")


def fig_qual_ignored(root):
    coco, objects, ignored = load_val(root)
    candidates = sorted(local_frames(root, coco), key=lambda i: -len(ignored[i["id"]]))
    seen, chosen = set(), []
    for info in candidates:
        if info["video"] not in seen and ignored[info["id"]]:
            seen.add(info["video"])
            chosen.append(info)
    chosen = chosen[:4]
    if not chosen:
        print("skip fig_qual_ignored: no local frame has ignored regions")
        return
    cols = 2
    fig, axes = plt.subplots((len(chosen) + 1) // 2, cols, figsize=(13, 3.9 * ((len(chosen) + 1) // 2)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, info in zip(axes.ravel(), chosen):
        image, k = thumb(root / "val" / info["file_name"], 900)
        draw(ax, image, [[v * k for v in a["bbox"]] for a in objects[info["id"]]], "#2fbf71")
        for a in ignored[info["id"]]:
            x, y, w, h = [v * k for v in a["bbox"]]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#e34948", linewidth=1.6))
        ax.set_title(f"{info['video']}: {len(objects[info['id']])} labelled, {len(ignored[info['id']])} ignored", loc="left", fontsize=9.5)
    finish(fig, "fig_qual_ignored_regions.png", "Labelled objects and ignored regions",
           "Crowds and other areas too packed to annotate are marked in red; detections there are not penalised", "Source: real validation frames.",
           legend=[Line2D([0], [0], color="#2fbf71", lw=2, label="labelled object"), Line2D([0], [0], color="#e34948", lw=2, label="ignored region")])


def fig_failures(folder, failures_json):
    images = sorted(Path(folder).glob("*.jpg"))[:3]
    if images:
        fig, axes = plt.subplots(len(images), 1, figsize=(11, 6.2 * len(images)))
        axes = np.atleast_1d(axes)
        for ax, path in zip(axes, images):
            ax.imshow(Image.open(path))
            ax.axis("off")
            ax.set_title(path.stem.split("_", 1)[1], loc="left", fontsize=9.5)
        finish(fig, "fig_failures_worst_frames.png", "The frames where the model fails most",
               "Green = found, red = missed, blue = false positive", f"Source: 07_failure_cases.py, run {Path(folder).name}.",
               legend=[Line2D([0], [0], color="lime", lw=2, label="found"), Line2D([0], [0], color="red", lw=2, label="missed"),
                       Line2D([0], [0], color="deepskyblue", lw=2, label="false positive")])
    if failures_json and Path(failures_json).is_file():
        report = json.loads(Path(failures_json).read_text())
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1, 1.3, 1.6]})
        for ax, key, title in ((axes[0], "recall_by_size", "Recall by object size"), (axes[1], "recall_by_class", "Recall by class")):
            data = report[key]
            names = [k for k in ("small", "medium", "large") if k in data] if key == "recall_by_size" else list(data)
            bars = ax.bar(range(len(names)), [data[n] for n in names], color=BASE, width=0.62)
            ax.set_xticks(range(len(names)), names, rotation=0 if key == "recall_by_size" else 40, fontsize=8.5)
            ax.set(ylim=(0, 1.05))
            ax.set_title(title, loc="left", fontsize=10.5)
            ax.grid(axis="x", visible=False)
            labels(ax, bars, "{:.0%}")
        pairs = report["top_class_confusions"][:6]
        bars = axes[2].barh(range(len(pairs)), [p["count"] for p in pairs], color=BASE, height=0.6)
        axes[2].set_yticks(range(len(pairs)), [f"{p['truth']} -> {p['predicted']}" for p in pairs])
        axes[2].invert_yaxis()
        axes[2].set_title("Most common class mix-ups (truth -> predicted)", loc="left", fontsize=10.5)
        axes[2].grid(axis="y", visible=False)
        labels(axes[2], bars, "{:d}", horizontal=True)
        small = report["settings"]["frames"] < 100
        finish(fig, "fig_failures_summary.png", "Where the model misses and what it confuses" + (" (preview, 17 frames)" if small else ""),
               f"Score >= {report['settings']['score_thr']}, IoU >= {report['settings']['iou']}, {report['settings']['frames']} frames", ("Source: 07_failure_cases.py. Rerun on all validation frames before quoting these numbers." if small else "Source: 07_failure_cases.py on all validation frames."))


# ---- the four models, scored the same way by stage 05 / 07 --------------------------------------
# the best run of each family under the shared protocol (see EXPERIMENTS.md); the Colab pilots are nb-*
EVAL_RUNS = {"frcnn-base": "Faster R-CNN", "retina-giou": "RetinaNet", "yolo26n-hires960": "YOLO26n", "rtdetr-l-e6": "RT-DETR-L"}

# every run of the tuning study: (family, run name, label); the first run of each family is its baseline
TUNING = [
    ("Faster R-CNN", "frcnn-base", "baseline"), ("Faster R-CNN", "frcnn-rpn300", "RPN proposals 2000 → 300"),
    ("Faster R-CNN", "frcnn-roi128", "RoI batch 512 → 128"),
    ("RetinaNet", "retina-base", "baseline"), ("RetinaNet", "retina-anchors16", "anchors 32-512 → 16-256"),
    ("RetinaNet", "retina-giou", "box loss L1 → GIoU"),
    ("YOLO26n", "yolo26n-base", "baseline (640 px)"), ("YOLO26n", "yolo26n-hires960", "input 640 → 960 px"),
    ("YOLO26n", "yolo26n-box10", "box loss weight 7.5 → 10"),
    ("RT-DETR-L", "rtdetr-l-e6", "baseline"), ("RT-DETR-L", "rtdetr-l-masked", "ignored regions masked"),
    ("RT-DETR-L", "rtdetr-l-e6-infer800", "inference at 800 px"), ("RT-DETR-L", "rtdetr-l-e6-infer960", "inference at 960 px"),
    ("Ensemble", "ens-rtdetr2", "2 RT-DETR-L models fused"), ("Ensemble", "ens-3models", "+ Faster R-CNN"),
    ("Ensemble", "ens-5models", "+ RetinaNet + YOLO26n"),
]


def eval_results(folder="eval"):
    found = {}
    for key, name in EVAL_RUNS.items():
        path = RESULTS / folder / f"{key}.json"
        if path.is_file():
            found[name] = json.loads(path.read_text())
    return found


def eval_note(res):
    return f"Source: results/eval/*.json, all {next(iter(res.values()))['frames']:,} validation frames, one protocol for every model."


def fig_eval_by_size():
    res = eval_results()
    if not res:
        return print("skip fig_eval_by_size: run 05_evaluate.py for the four models first")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(3)
    width = 0.8 / len(res)
    for k, (name, r) in enumerate(res.items()):
        values = [r[key] or 0 for key in ("AP_small", "AP_medium", "AP_large")]
        bars = ax.bar(x + (k - (len(res) - 1) / 2) * width, values, width * 0.92, color=MODEL_COLOUR[name], label=name)
        labels(ax, bars, "{:.2f}")
    ax.set_xticks(x, ["small  (< 32 px)", "medium  (32 - 96 px)", "large  (> 96 px)"])
    ax.set(ylabel="AP (higher is better)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(res), loc="upper left")
    finish(fig, "fig_eval_by_size.png", "Detection quality by object size",
           "Every model is far weaker on small objects", eval_note(res))


def fig_eval_speed():
    res = eval_results()
    if not res:
        return print("skip fig_eval_speed")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for name, r in res.items():
        ax.scatter(r["ms_per_frame"], r["AP"], s=90, color=MODEL_COLOUR[name], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"{name}\n{r['AP']:.3f} AP, {r['ms_per_frame']:.0f} ms", (r["ms_per_frame"], r["AP"]), xytext=(10, 14 if name == "Faster R-CNN" else -14 if name == "RetinaNet" else -4),
                    textcoords="offset points", fontsize=9, va="center")
    ax.set(xscale="log", xlabel="inference time per frame, ms (log scale, lower is faster)", ylabel="AP (higher is better)")
    ax.set_xlim(left=min(r["ms_per_frame"] for r in res.values()) * 0.6, right=max(r["ms_per_frame"] for r in res.values()) * 4)
    ax.set_ylim(0, max(r["AP"] for r in res.values()) * 1.2)
    finish(fig, "fig_eval_speed_accuracy.png", "Accuracy against speed",
           "Top left is best: accurate and fast", eval_note(res))


def heat(ax, matrix, rows, cols, fmt="{:.2f}"):
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(0.01, np.nanmax(matrix)) * 1.15, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = matrix[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else fmt.format(v), ha="center", va="center", fontsize=9,
                    color="white" if v > 0.6 * np.nanmax(matrix) else INK)


def fig_eval_per_class():
    res = eval_results()
    if not res:
        return print("skip fig_eval_per_class")
    matrix = np.array([[res[m]["per_class"].get(c) if res[m]["per_class"].get(c) is not None else np.nan for c in CLASS_NAMES] for m in res])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    heat(ax, matrix, list(res), CLASS_NAMES)
    finish(fig, "fig_eval_per_class.png", "AP by class and model", "Darker = higher", eval_note(res))


def fig_eval_per_clip():
    res = {n: r for n, r in eval_results().items() if "per_clip" in r}
    if not res:
        return print("skip fig_eval_per_clip: run 05_evaluate.py with --per-clip")
    best = max(res, key=lambda n: res[n]["AP"])
    clips = sorted(res[best]["per_clip"], key=lambda c: res[best]["per_clip"][c]["AP"] or 0)
    frames = {c: res[best]["per_clip"][c]["frames"] for c in clips}
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for name, r in res.items():
        ax.scatter([r["per_clip"][c]["AP"] or 0 for c in clips], range(len(clips)), s=60, color=MODEL_COLOUR[name], edgecolor=SURFACE, linewidth=1.2, label=name, zorder=3)
    ax.set_yticks(range(len(clips)), [f"{c}  ({frames[c]} frames)" for c in clips])
    ax.set(xlabel="AP on the clip (higher is better)")
    ax.legend(loc="lower right")
    finish(fig, "fig_eval_per_clip.png", "Every model, clip by clip",
           "Scores differ more between clips than between the three weaker models", eval_note(res))


def fig_failure_compare():
    res = eval_results("failures")
    if not res:
        return print("skip fig_failure_compare: run 07_failure_cases.py for the four models first")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    x = np.arange(3)
    width = 0.8 / len(res)
    for k, (name, r) in enumerate(res.items()):
        bars = ax.bar(x + (k - (len(res) - 1) / 2) * width, [r["recall_by_size"].get(b, 0) for b in ("small", "medium", "large")], width * 0.92, color=MODEL_COLOUR[name], label=name)
        labels(ax, bars, "{:.0%}", pad=2)
    ax.set_xticks(x, ["small", "medium", "large"])
    ax.set(ylim=(0, 1.1))
    ax.set_title("Share of objects found, by size", loc="left", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    matrix = np.array([[r["recall_by_class"].get(c, np.nan) for c in CLASS_NAMES] for r in res.values()])
    heat(axes[1], matrix, list(res), CLASS_NAMES, "{:.0%}")
    axes[1].set_title("Share of objects found, by class", loc="left", fontsize=10.5)
    first = next(iter(res.values()))["settings"]
    finish(fig, "fig_failure_compare.png", "What each model misses", f"Score >= {first['score_thr']}, IoU >= {first['iou']}, {first['frames']:,} validation frames",
           "Source: results/failures/*.json from 07_failure_cases.py.")




def fig_tuning():
    rows = [(f, n, label, json.loads((RESULTS / "eval" / f"{n}.json").read_text()))
            for f, n, label in TUNING if (RESULTS / "eval" / f"{n}.json").is_file()]
    if not rows:
        return print("skip fig_tuning: no tuning runs scored yet")
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.6))
    y = np.arange(len(rows))[::-1]
    base = {}
    for f, n, label, r in rows:
        base.setdefault(f, r["AP"])
    base["Ensemble"] = next((r["AP"] for f, n, _, r in rows if n == "rtdetr-l-e6"), base.get("Ensemble"))   # against the best single model
    bars = ax.barh(y, [r["AP"] for *_, r in rows], color=[MODEL_COLOUR[f] for f, *_ in rows], height=0.62)
    for bar, (f, n, label, r), yy in zip(bars, rows, y):
        delta = r["AP"] - base[f]
        text = f"{r['AP']:.3f}" if f != "Ensemble" and n == next(k for ff, k, _ in TUNING if ff == f) else f"{r['AP']:.3f}  ({delta:+.3f})"
        ax.text(r["AP"] + 0.003, yy, text, va="center", fontsize=9)
    ax.set_yticks(y, [f"{f}  ·  {label}" for f, n, label, r in rows])
    ax.set(xlim=(0, 0.31), xlabel="AP on all 2,846 validation frames (higher is better); brackets: change against the baseline")
    ax.grid(axis="y", visible=False)
    finish(fig, "fig_tuning.png", "What each change did to the score",
           "One change per run against its family's baseline; ensembles against the best single model",
           "Source: results/eval/*.json. One protocol for every run. Differences under about 0.005 AP are within run-to-run noise.")


def fig_equal_budget():
    curves = []
    for f, n, label in (("Faster R-CNN", "frcnn-base", ""), ("RetinaNet", "retina-base", "")):
        h = json.loads((RESULTS / "runs" / n / "history.json").read_text())
        curves.append((f, [0] + [e["minutes"] for e in h], [0] + [e["val_AP"] for e in h], "-"))
    for f, n, style in (("YOLO26n", "yolo26n-base", "-"), ("RT-DETR-L", "rtdetr-l-masked", "-"), ("RT-DETR-L", "rtdetr-l-e6", "--")):
        path = RESULTS / "runs" / n / "results.csv"
        if not path.is_file():
            continue
        rows = list(csv.DictReader(path.open()))
        curves.append((f if style == "-" else f + " (MacBook run)", [0] + [float(r["time"]) / 60 for r in rows],
                       [0] + [float(r["metrics/mAP50-95(B)"]) for r in rows], style))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name, x, yv, style in curves:
        ax.plot(x, yv, style, marker="o", ms=4, lw=2, color=MODEL_COLOUR[name.split(" (")[0]], label=name)
    ax.set(xlabel="training time, minutes", ylabel="quick-validation AP (206 frames)", xlim=(0, 50))
    ax.legend(loc="lower right")
    finish(fig, "fig_equal_budget.png", "Accuracy over training time",
           "Query-based RT-DETR-L against the CNN detectors on the same machine (Mac Mini)",
           "Source: results/runs/*. Quick checks on a validation subset (not final scores). The dashed MacBook run continues to 95 min.")


GROUPS = {
    "results": [fig_progress, fig_models, fig_rtdetr_training, fig_quick_validation, fig_per_class],
    "eval": [fig_tuning, fig_equal_budget, fig_eval_by_size, fig_eval_speed, fig_eval_per_class, fig_eval_per_clip, fig_failure_compare],
    "data": [fig_data_classes, fig_data_sizes, fig_data_density, fig_data_resolution, fig_data_ignored, fig_data_stride],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["results", "data", "eval", "frames"], help="draw one group")
    parser.add_argument("--data", type=Path)
    run = "rtdetr-l-e6"
    parser.add_argument("--predictions", type=Path, default=REPO / "runs" / "eval" / f"{run}_predictions.json")
    parser.add_argument("--failures", type=Path, default=REPO / "runs" / "failures" / run, help="folder of overlays from stage 07")
    parser.add_argument("--failures-json", type=Path, default=RESULTS / "failures" / f"{run}.json")
    args = parser.parse_args()
    for group, functions in GROUPS.items():
        if args.only in (None, group):
            for function in functions:
                function()
    if args.only in (None, "frames"):
        root = data_root(args.data)
        if args.predictions.is_file():
            fig_qual_predictions(root, args.predictions)
        else:
            print("skip fig_qual_predictions: run 05_evaluate.py --save-predictions first")
        fig_qual_ignored(root)
        if args.failures.is_dir():
            fig_failures(args.failures, args.failures_json)
        else:
            print("skip failure figures: run 07_failure_cases.py first")


if __name__ == "__main__":
    main()
