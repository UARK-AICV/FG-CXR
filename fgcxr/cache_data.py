from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .data import _resize_tensor, load_manifest
from torchvision.transforms import InterpolationMode


def _save_image(tensor, path: Path) -> None:
    array = (tensor.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    Image.fromarray(array).save(temporary, format="JPEG", quality=95)
    os.replace(temporary, path)


def _cache_one(payload: tuple[str, dict, str]) -> str:
    data_root_raw, study, cache_root_raw = payload
    data_root = Path(data_root_raw)
    cache_root = Path(cache_root_raw)
    source_image = Image.open(data_root / study["image"]).convert("RGB")
    for size in (224, 384):
        output = cache_root / f"images_{size}" / f"{study['dicom_id']}.jpg"
        if not output.is_file():
            _save_image(_resize_tensor(source_image, size, InterpolationMode.BICUBIC), output)

    heatmap_output = cache_root / "heatmaps_224" / f"{study['study_id']}.npz"
    if not heatmap_output.is_file():
        heatmaps = np.stack(
            [
                (_resize_tensor(Image.open(data_root / region["heatmap"]).convert("L"), 224, InterpolationMode.BILINEAR)[0].numpy() * 255)
                .round()
                .clip(0, 255)
                .astype(np.uint8)
                for region in study["regions"]
            ]
        )
        temporary = heatmap_output.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, heatmaps=heatmaps)
        os.replace(temporary, heatmap_output)
    return study["study_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache resized FG-CXR tensors for fast local training")
    parser.add_argument("--manifest", type=Path, default=Path("data/fg_cxr/manifest.json"))
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root, manifest = load_manifest(args.manifest)
    cache_root = args.cache_root or data_root / "cache"
    for directory in (cache_root / "images_224", cache_root / "images_384", cache_root / "heatmaps_224"):
        directory.mkdir(parents=True, exist_ok=True)
    payloads = [(str(data_root), study, str(cache_root)) for study in manifest["studies"]]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(_cache_one, payloads, chunksize=4), total=len(payloads), desc="cache FG-CXR"))
    print(f"Cached {len(payloads)} studies under {cache_root}")


if __name__ == "__main__":
    main()
