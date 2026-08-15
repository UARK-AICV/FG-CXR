from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import REGIONS
from .data import HeatmapDataset, PredictedHeatmaps, ReportDataset, assert_region_order, load_manifest, report_collate, select_studies
from .models.report import build_report_model
from .utils import choose_device, write_json


def caption_metrics(predictions: list[dict[str, str]]) -> dict[str, float]:
    # Match the recovered COCOCaptionMetrics preprocessing exactly.
    normalize = lambda value: re.sub(" +", " ", value.replace(".", " ."))
    references = {item["id"]: [normalize(item["reference"])] for item in predictions}
    hypotheses = {item["id"]: [normalize(item["prediction"])] for item in predictions}
    metrics: dict[str, float] = {}
    scorers = ((Bleu(4), ("bleu_1", "bleu_2", "bleu_3", "bleu_4")), (Meteor(), ("meteor",)), (Rouge(), ("rouge",)), (Cider(), ("cider",)))
    for scorer, names in scorers:
        score, _ = scorer.compute_score(references, hypotheses)
        values = score if isinstance(score, (list, tuple)) else (score,)
        metrics.update({name: float(value) for name, value in zip(names, values, strict=True)})

    sentence_diversity: list[float] = []
    for item in predictions:
        words = item["prediction"].lower().replace(",", " ").rstrip(". ").split()
        pairs = list(zip(words, words[1:]))
        sentence_diversity.append(len(set(pairs)) / max(len(words), 1))
    # This is the recovered evaluator's per-sentence Div@2 definition, not the
    # corpus-global distinct-2 statistic used by some captioning repositories.
    metrics["div_2"] = float(np.mean(sentence_diversity))
    metrics["count"] = len(predictions)
    return metrics


@torch.inference_mode()
def evaluate_reports(args: argparse.Namespace) -> dict[str, float]:
    device = choose_device(args.device)
    data_root, manifest = load_manifest(args.manifest)
    assert_region_order(manifest)
    predicted = PredictedHeatmaps(args.predicted_heatmaps) if args.predicted_heatmaps else None
    studies = select_studies(manifest, "test")
    if args.max_test_studies is not None:
        studies = studies[: args.max_test_studies]
    loader = DataLoader(
        ReportDataset(data_root, studies, predicted),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=report_collate,
    )
    model, tokenizer = build_report_model()
    state = torch.load(args.report_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    model.decoder.config.use_cache = True
    predictions: list[dict[str, str]] = []
    for batch in tqdm(loader, desc="generate reports"):
        generated = model.generate(
            batch["images"].to(device, non_blocking=True),
            batch["heatmaps"].to(device, non_blocking=True),
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            args.max_new_tokens,
            args.num_beams,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for batch_index, study_id in enumerate(batch["study_ids"]):
            for region_index, region in enumerate(REGIONS):
                predictions.append({
                    "id": f"{study_id}_{region.replace(' ', '_')}",
                    "study_id": study_id,
                    "region": region,
                    "reference": batch["reports"][batch_index][region_index],
                    "prediction": decoded[batch_index * len(REGIONS) + region_index].strip(),
                })
    write_json(predictions, args.output_dir / "predictions.json")
    metrics = caption_metrics(predictions)
    write_json(metrics, args.output_dir / "report_metrics.json")
    return metrics


def heatmap_metrics(args: argparse.Namespace) -> dict[str, float]:
    data_root, manifest = load_manifest(args.manifest)
    studies = select_studies(manifest, "test")
    if args.max_test_studies is not None:
        studies = studies[: args.max_test_studies]
    ground_truth = HeatmapDataset(data_root, studies)
    predicted = PredictedHeatmaps(args.predicted_heatmaps)
    confusion = np.zeros((2, 2), dtype=np.int64)
    l1_values: list[float] = []
    l2_values: list[float] = []
    ssim_values: list[float] = []
    psnr_values: list[float] = []
    for item in tqdm(ground_truth, desc="evaluate heatmaps"):
        target = item["heatmaps"]
        prediction = predicted.get(item["study_id"])
        prediction = F.interpolate(prediction[:, None], size=target.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        prediction = prediction / prediction.flatten(1).amax(1).clamp_min(1e-6)[:, None, None]
        target = target / target.flatten(1).amax(1).clamp_min(1e-6)[:, None, None]
        target_masks = item["masks"].bool()
        prediction_masks = prediction > 0.5
        truth_flat = target_masks.flatten().numpy().astype(np.int64)
        guess_flat = prediction_masks.flatten().numpy().astype(np.int64)
        confusion += np.bincount(2 * guess_flat + truth_flat, minlength=4).reshape(2, 2)
        for region_index in range(len(REGIONS)):
            truth_np = target[region_index].numpy()
            pred_np = prediction[region_index].numpy()
            l1_values.append(float(np.mean(np.abs(pred_np - truth_np))))
            l2_values.append(float(np.mean((pred_np - truth_np) ** 2)))
            ssim_values.append(float(structural_similarity(truth_np, pred_np, data_range=1.0)))
            psnr_values.append(float(peak_signal_noise_ratio(truth_np, pred_np, data_range=1.0)))
    true_positive = np.diag(confusion).astype(float)
    positive_truth = confusion.sum(axis=0)
    positive_prediction = confusion.sum(axis=1)
    iou = true_positive / np.maximum(positive_truth + positive_prediction - true_positive, 1)
    weights = positive_truth / positive_truth.sum()
    metrics = {
        "fg_iou": float(iou[1] * 100),
        "bg_iou": float(iou[0] * 100),
        "fw_iou": float((iou * weights).sum() * 100),
        "ssim": float(np.mean(ssim_values)),
        "psnr": float(np.mean(psnr_values)),
        "l1": float(np.mean(l1_values)),
        "l2": float(np.mean(l2_values)),
    }
    write_json(metrics, args.output_dir / "heatmap_metrics.json")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FG-CXR heatmaps and region reports")
    parser.add_argument("--manifest", type=Path, default=Path("data/fg_cxr/manifest.json"))
    parser.add_argument("--predicted-heatmaps", type=Path, required=True)
    parser.add_argument("--report-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation_2k"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-test-studies", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--skip-heatmaps", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    if not args.skip_heatmaps:
        results["heatmaps"] = heatmap_metrics(args)
    if args.report_checkpoint:
        results["reports"] = evaluate_reports(args)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
