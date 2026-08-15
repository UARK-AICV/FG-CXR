from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .constants import REGIONS
from .data import PredictedHeatmaps, ReportDataset, assert_region_order, load_manifest, report_collate, select_studies
from .models.report import build_report_model
from .utils import atomic_torch_save, choose_device, parameter_count, seed_everything


def tokenize_reports(tokenizer, reports: list[list[str]], max_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = [report for study_reports in reports for report in study_reports]
    encoded = tokenizer(
        flat,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - 2,
    )["input_ids"]
    sequences = [
        [tokenizer.bos_token_id, *sequence, tokenizer.eos_token_id]
        for sequence in encoded
    ]
    length = min(max(len(sequence) for sequence in sequences), max_length)
    input_ids = torch.full((len(sequences), length), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        sequence = sequence[:length]
        input_ids[row, : len(sequence)] = torch.tensor(sequence)
        attention[row, : len(sequence)] = 1
    labels = input_ids.clone()
    labels[attention == 0] = -100
    shape = (len(reports), len(REGIONS), length)
    return input_ids.view(shape).to(device), attention.view(shape).to(device), labels.view(shape).to(device)


@torch.inference_mode()
def validate(model, tokenizer, loader: DataLoader, device: torch.device, max_length: int, amp: bool) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        input_ids, attention, labels = tokenize_reports(tokenizer, batch["reports"], max_length, device)
        images = batch["images"].to(device, non_blocking=True)
        heatmaps = batch["heatmaps"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = model(images, heatmaps, input_ids, attention, labels)
        total += float(loss) * images.shape[0]
        count += images.shape[0]
    return total / max(count, 1)


def checkpoint_payload(model, optimizer, scaler, step: int, best: float, args: argparse.Namespace) -> dict:
    return {
        "format": "fgcxr-report-v1",
        "step": step,
        "best_val_loss": best,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CvT-21 + DistilGPT-2 on region-masked FG-CXR features")
    parser.add_argument("--manifest", type=Path, default=Path("data/fg_cxr/manifest.json"))
    parser.add_argument("--predicted-heatmaps", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/report_2k"))
    parser.add_argument("--max-train-studies", type=int, default=2000)
    parser.add_argument("--max-val-studies", type=int)
    parser.add_argument("--max-steps", type=int, default=6000)
    # Four studies contain 28 region/report pairs, close to the paper's batch 32
    # while using only ~3.5 GB in the verified RTX 3080 environment.
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    data_root, manifest = load_manifest(args.manifest)
    assert_region_order(manifest)
    predicted = PredictedHeatmaps(args.predicted_heatmaps) if args.predicted_heatmaps else None
    train_studies = select_studies(manifest, "train", args.max_train_studies, args.seed)
    val_studies = select_studies(manifest, "val", None, args.seed)
    if args.max_val_studies is not None:
        val_studies = val_studies[: args.max_val_studies]
    train_loader = DataLoader(
        ReportDataset(data_root, train_studies, predicted),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=report_collate,
    )
    val_loader = DataLoader(
        ReportDataset(data_root, val_studies, predicted),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=report_collate,
    )
    model, tokenizer = build_report_model()
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    model.decoder.gradient_checkpointing_enable()
    model.decoder.config.use_cache = False
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    best = math.inf
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step = int(state["step"])
        best = float(state["best_val_loss"])

    print(json.dumps({
        "device": str(device),
        "heatmaps": "predicted" if predicted else "ground_truth",
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
    micro_since_step = 0
    while step < args.max_steps:
        epoch += 1
        for batch in train_loader:
            input_ids, attention, labels = tokenize_reports(tokenizer, batch["reports"], args.max_length, device)
            images = batch["images"].to(device, non_blocking=True)
            heatmaps = batch["heatmaps"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = model(images, heatmaps, input_ids, attention, labels)
                scaled_loss = loss / args.accumulate
            scaler.scale(scaled_loss).backward()
            micro_since_step += 1
            if micro_since_step < args.accumulate:
                continue
            micro_since_step = 0
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
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
                    f"step={step}/{args.max_steps} epoch={epoch} loss={loss.detach().item():.4f} "
                    f"steps/s={rate:.3f} eta_min={eta / 60:.1f}",
                    flush=True,
                )
            if step % args.val_every == 0 or step == args.max_steps:
                value = validate(model, tokenizer, val_loader, device, args.max_length, amp)
                print(f"validation step={step} loss={value:.6f}", flush=True)
                payload = checkpoint_payload(model, optimizer, scaler, step, min(best, value), args)
                atomic_torch_save(payload, args.output_dir / "latest.pt")
                if value < best:
                    best = value
                    atomic_torch_save(payload, args.output_dir / "best.pt")
                model.train()
                model.decoder.config.use_cache = False
            if step >= args.max_steps:
                break


if __name__ == "__main__":
    main()
