#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m fgcxr.train_heatmap \
  --max-train-studies 2000 \
  --batch-size 8 --max-steps 6000 --output-dir outputs/iai_2k

"$PYTHON_BIN" -m fgcxr.predict_heatmaps \
  --checkpoint outputs/iai_2k/best.pt \
  --output outputs/iai_2k/predicted_heatmaps.npz --split all --batch-size 16

"$PYTHON_BIN" -m fgcxr.train_report \
  --max-train-studies 2000 \
  --predicted-heatmaps outputs/iai_2k/predicted_heatmaps.npz \
  --batch-size 4 --max-steps 6000 --output-dir outputs/report_2k

"$PYTHON_BIN" -m fgcxr.evaluate \
  --predicted-heatmaps outputs/iai_2k/predicted_heatmaps.npz \
  --report-checkpoint outputs/report_2k/latest.pt \
  --output-dir outputs/evaluation_2k
