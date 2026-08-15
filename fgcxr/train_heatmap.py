from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import HeatmapDataset, assert_region_order, load_manifest, select_studies
from .models.iai import build_iai_model, heatmap_loss
from .utils import atomic_torch_save, choose_device, parameter_count, seed_everything, trainable_state_dict


@torch.inference_mode()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "l1": 0.0, "l2": 0.0, "fg_iou": 0.0, "count": 0}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["heatmaps"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            outputs = model(images)
            losses = heatmap_loss(outputs, targets, masks)
        logits = outputs["logits"]
        predictions = logits.sigmoid()
        targets_small = F.interpolate(targets, size=predictions.shape[-2:], mode="bilinear", align_corners=False)
        masks_small = F.interpolate(masks, size=predictions.shape[-2:], mode="nearest").bool()
        predicted_masks = predictions > 0.5
        intersection = (predicted_masks & masks_small).flatten(2).sum(-1)
        union = (predicted_masks | masks_small).flatten(2).sum(-1).clamp_min(1)
        count = images.shape[0]
        totals["loss"] += float(losses.total) * count
        totals["l1"] += float(F.l1_loss(predictions, targets_small)) * count
        totals["l2"] += float(F.mse_loss(predictions, targets_small)) * count
        totals["fg_iou"] += float((intersection / union).mean()) * count
        totals["count"] += count
    count = max(1, totals.pop("count"))
    return {key: value / count for key, value in totals.items()}


def checkpoint_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, step: int, best: float, args: argparse.Namespace) -> dict:
    return {
        "format": "fgcxr-iai-v1",
        "step": step,
        "best_val_loss": best,
        "model": trainable_state_dict(model, "visual."),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FG-CXR IAI heatmap predictor")
    parser.add_argument("--manifest", type=Path, default=Path("data/fg_cxr/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/iai_2k"))
    parser.add_argument("--max-train-studies", type=int, default=2000)
    parser.add_argument("--max-val-studies", type=int)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    data_root, manifest = load_manifest(args.manifest)
    assert_region_order(manifest)
    train_studies = select_studies(manifest, "train", args.max_train_studies, args.seed)
    val_studies = select_studies(manifest, "val", None, args.seed)
    if args.max_val_studies is not None:
        val_studies = val_studies[: args.max_val_studies]
    train_loader = DataLoader(
        HeatmapDataset(data_root, train_studies, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        HeatmapDataset(data_root, val_studies),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_iai_model().to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    best = math.inf
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step = int(state["step"])
        best = float(state["best_val_loss"])

    print(json.dumps({
        "device": str(device),
        "train_studies": len(train_studies),
        "val_studies": len(val_studies),
        "effective_region_batch": args.batch_size * args.accumulate * 7,
        "parameters_m": round(parameter_count(model) / 1e6, 2),
        "trainable_parameters_m": round(parameter_count(model, True) / 1e6, 2),
    }, indent=2))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    epoch = 0
    while step < args.max_steps:
        epoch += 1
        for micro_step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["heatmaps"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                outputs = model(images)
                losses = heatmap_loss(outputs, targets, masks)
                scaled_loss = losses.total / args.accumulate
            scaler.scale(scaled_loss).backward()
            if micro_step % args.accumulate:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % 20 == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = max(time.monotonic() - start, 1e-6)
                rate = step / elapsed
                eta = (args.max_steps - step) / max(rate, 1e-6)
                print(
                    f"step={step}/{args.max_steps} epoch={epoch} loss={losses.total.detach().item():.4f} "
                    f"l2={losses.l2.detach().item():.4f} dice={losses.dice.detach().item():.4f} "
                    f"steps/s={rate:.3f} eta_min={eta / 60:.1f}",
                    flush=True,
                )
            if step % args.val_every == 0 or step == args.max_steps:
                metrics = validate(model, val_loader, device, amp)
                print(f"validation step={step} {json.dumps(metrics, sort_keys=True)}", flush=True)
                payload = checkpoint_payload(model, optimizer, scaler, step, min(best, metrics["loss"]), args)
                atomic_torch_save(payload, args.output_dir / "latest.pt")
                if metrics["loss"] < best:
                    best = metrics["loss"]
                    atomic_torch_save(payload, args.output_dir / "best.pt")
                model.train()
            if step >= args.max_steps:
                break


if __name__ == "__main__":
    main()
