from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import HeatmapDataset, assert_region_order, load_manifest, select_studies
from .models.iai import build_iai_model
from .utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict all seven FG-CXR gaze heatmaps")
    parser.add_argument("--manifest", type=Path, default=Path("data/fg_cxr/manifest.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/iai_2k/predicted_heatmaps.npz"))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--feature-size", type=int, default=24)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    data_root, manifest = load_manifest(args.manifest)
    assert_region_order(manifest)
    studies = select_studies(manifest, args.split)
    loader = DataLoader(
        HeatmapDataset(data_root, studies),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = build_iai_model()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if unexpected or any(not key.startswith("visual.") for key in missing):
        raise RuntimeError(f"Checkpoint mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}")
    model.to(device).eval()
    all_ids: list[str] = []
    all_heatmaps: list[np.ndarray] = []
    amp = device.type == "cuda"
    with torch.inference_mode():
        for batch in tqdm(loader, desc="predict heatmaps"):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(batch["image"].to(device, non_blocking=True))["logits"]
                heatmaps = F.interpolate(
                    logits.sigmoid(), size=(args.feature_size, args.feature_size), mode="bilinear", align_corners=False
                )
            all_ids.extend(batch["study_id"])
            all_heatmaps.append(heatmaps.cpu().half().numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        study_ids=np.asarray(all_ids),
        heatmaps=np.concatenate(all_heatmaps),
        regions=np.asarray(manifest["regions"]),
    )
    print(f"Wrote {len(all_ids)} x 7 heatmaps to {args.output}")


if __name__ == "__main__":
    main()
