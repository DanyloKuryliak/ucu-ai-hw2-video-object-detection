"""Stage 03: train Faster R-CNN (two-stage) or RetinaNet (one-stage, anchor-based) with TorchVision.

Both start from COCO-pretrained weights and are fine-tuned on a per-clip frame sample. A quick validation
on every Nth val frame runs every --eval-every steps; the best checkpoint is kept and training stops after
--patience checks without improvement, or at --max-steps / --max-minutes. The per-check history
(results/runs/<name>/history.json) is the mAP-over-training curve used to compare detectors at equal budget.

Examples (one change at a time, so each effect is attributable):
  python scripts/03_train_torchvision.py fasterrcnn --name frcnn-base
  python scripts/03_train_torchvision.py fasterrcnn --name frcnn-roi128 --box-batch-size-per-image 128
  python scripts/03_train_torchvision.py retinanet --name retina-giou --box-loss giou
  python scripts/03_train_torchvision.py retinanet --name retina-anchors16 --anchor-base 16,32,64,128,256
"""

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import RESULTS, RUNS, data_root, load_coco, pick_device, sample_frames, save_json, seed_everything
from data import DetectionDataset, collate
from detectors import DEFAULTS, build_torchvision, load_checkpoint, save_checkpoint, torchvision_detector
from evaluation import coco_evaluate, load_gt, run_detector


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("family", choices=["fasterrcnn", "retinanet"])
    p.add_argument("--name", required=True, help="run name; outputs go to runs/<name> and results/runs/<name>")
    p.add_argument("--data", type=Path)
    p.add_argument("--stride", type=int, default=5, help="train on every Nth frame of each clip")
    p.add_argument("--mask-ignored", action="store_true", help="paint ignored regions grey in train images")
    p.add_argument("--flip", action="store_true", help="random horizontal flip")
    p.add_argument("--no-pretrained", action="store_true", help="skip the COCO weights (smoke tests only)")
    # optimisation
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--schedule", choices=["cosine", "step", "const"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    # budget and validation
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--max-minutes", type=float, default=0, help="wall-clock budget; 0 = unlimited")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--val-stride", type=int, default=14, help="quick validation on every Nth val frame")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=0.002)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device")
    p.add_argument("--resume", type=Path, help="last.pt of an earlier run with the same name")
    # model knobs
    p.add_argument("--min-size", type=int, default=DEFAULTS["min_size"], help="short side the model resizes to")
    p.add_argument("--max-size", type=int, default=DEFAULTS["max_size"])
    p.add_argument("--rpn-pre-nms-top-n-train", type=int, default=DEFAULTS["rpn_pre_nms_top_n_train"])
    p.add_argument("--rpn-post-nms-top-n-train", type=int, default=DEFAULTS["rpn_post_nms_top_n_train"], help="proposals kept per image")
    p.add_argument("--box-batch-size-per-image", type=int, default=DEFAULTS["box_batch_size_per_image"], help="RoIs sampled per image")
    p.add_argument("--box-positive-fraction", type=float, default=DEFAULTS["box_positive_fraction"], help="share of positive RoIs")
    p.add_argument("--anchor-base", default=",".join(map(str, DEFAULTS["anchor_base"])), help="RetinaNet size per FPN level")
    p.add_argument("--aspect-ratios", default=",".join(map(str, DEFAULTS["aspect_ratios"])))
    p.add_argument("--fg-iou", type=float, default=DEFAULTS["fg_iou"], help="RetinaNet positive IoU threshold")
    p.add_argument("--bg-iou", type=float, default=DEFAULTS["bg_iou"], help="RetinaNet negative IoU threshold")
    p.add_argument("--focal-alpha", type=float, default=DEFAULTS["focal_alpha"])
    p.add_argument("--focal-gamma", type=float, default=DEFAULTS["focal_gamma"])
    p.add_argument("--box-loss", choices=["l1", "smooth_l1", "giou", "diou", "ciou"], default=DEFAULTS["box_loss"], help="RetinaNet box loss")
    return p.parse_args()


def model_config(args) -> dict:
    return {"min_size": args.min_size, "max_size": args.max_size,
            "rpn_pre_nms_top_n_train": args.rpn_pre_nms_top_n_train, "rpn_post_nms_top_n_train": args.rpn_post_nms_top_n_train,
            "box_batch_size_per_image": args.box_batch_size_per_image, "box_positive_fraction": args.box_positive_fraction,
            "anchor_base": [int(x) for x in args.anchor_base.split(",")], "aspect_ratios": [float(x) for x in args.aspect_ratios.split(",")],
            "fg_iou": args.fg_iou, "bg_iou": args.bg_iou, "focal_alpha": args.focal_alpha, "focal_gamma": args.focal_gamma,
            "box_loss": args.box_loss}


def lr_factor(args, step: int) -> float:
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    if args.schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return 0.1 if args.schedule == "step" and progress > 0.75 else 1.0


def main():
    args = parse()
    seed_everything(args.seed)
    device = pick_device(args.device)
    root = data_root(args.data)
    run_dir, result_dir = RUNS / args.name, RESULTS / "runs" / args.name
    cfg = model_config(args)
    save_json(result_dir / "config.json", {"args": {k: str(v) for k, v in vars(args).items()}, "model": cfg})

    dataset = DetectionDataset(root, "train", args.stride, -1 if args.family == "retinanet" else 0, args.mask_ignored, args.flip)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True,
                        num_workers=args.workers, generator=generator, persistent_workers=args.workers > 0)
    val_gt = load_gt(root / "val" / "annotations_coco.json")
    val_images = sample_frames(load_coco(root, "val")["images"], args.val_stride)
    print(f"{args.family} on {device}: {len(dataset)} train frames, {len(val_images)} quick-validation frames", flush=True)

    model = build_torchvision(args.family, cfg, pretrained=not args.no_pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = (torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay) if args.optimizer == "sgd"
                 else torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(args, s))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.startswith("cuda"))
    step, best, stale, history, spent = 0, -1.0, 0, [], 0.0
    if args.resume:
        saved = load_checkpoint(args.resume)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step, best, stale, history, spent = saved["step"], saved["best"], saved["stale"], saved["history"], saved["minutes"]
        scheduler.last_epoch = step - 1
        print(f"resumed at step {step}", flush=True)

    started, recent, bad_steps = time.perf_counter(), [], 0
    model.train()
    stop = False
    while not stop:
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                bad_steps += 1
                assert bad_steps < 10, "loss is not finite for 10 steps in a row; lower --lr"
                optimizer.zero_grad(set_to_none=True)
                continue
            bad_steps = 0
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            recent.append(loss.item())
            minutes = spent + (time.perf_counter() - started) / 60
            if step % 50 == 0:
                print(f"step {step} | loss {sum(recent[-50:]) / len(recent[-50:]):.4f} | {minutes:.1f} min", flush=True)
            out_of_budget = step >= args.max_steps or (args.max_minutes and minutes >= args.max_minutes)
            if step % args.eval_every == 0 or out_of_budget:
                model.eval()
                detections, ms = run_detector(torchvision_detector(model, args.family, device), root, "val", val_images, device, log_every=0)
                metrics = coco_evaluate(val_gt, detections, [i["id"] for i in val_images], per_class=False)
                model.train()
                history.append({"step": step, "minutes": round(minutes, 2), "train_loss": round(sum(recent[-args.eval_every:]) / len(recent[-args.eval_every:]), 4),
                                "val_AP": metrics["AP"], "val_AP50": metrics["AP50"], "val_AP_small": metrics["AP_small"]})
                improved = metrics["AP"] >= best + args.min_delta
                if improved:
                    best, stale = metrics["AP"], 0
                    save_checkpoint(run_dir / "best.pt", args.family, cfg, model, step=step, val_AP=best)
                else:
                    stale += 1
                save_checkpoint(run_dir / "last.pt", args.family, cfg, model, optimizer=optimizer.state_dict(),
                                step=step, best=best, stale=stale, history=history, minutes=minutes)
                save_json(result_dir / "history.json", history)
                print(f"  check @ {step}: val AP {metrics['AP']:.4f} (best {best:.4f}, no-gain {stale}/{args.patience})", flush=True)
                if stale >= args.patience:
                    print("early stop: no improvement")
                    out_of_budget = True
            if out_of_budget:
                stop = True
                break
    summary = {"best_val_AP": best, "steps": step, "minutes": round(spent + (time.perf_counter() - started) / 60, 2), "best_checkpoint": str(run_dir / "best.pt")}
    save_json(result_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
