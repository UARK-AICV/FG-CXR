# FG-CXR reproduction

Public training, inference, and evaluation code for **FG-CXR: A
Radiologist-Aligned Gaze Dataset for Enhancing Interpretability in Chest X-Ray
Report Generation** ([paper](https://arxiv.org/abs/2411.15413),
[dataset](https://huggingface.co/datasets/phamtrongthang/FG-CXR-dataset)).

The pipeline has two stages:

1. An IAI-style side adapter predicts seven anatomy-conditioned gaze heatmaps
   from intermediate BiomedCLIP features.
2. CvT-21 produces image features, the predicted regional heatmaps mask those
   features, and DistilGPT-2 generates one sentence for each anatomical region.

The implementation uses current PyTorch and predicts all seven fixed anatomy
queries together. It preserves the model interface and losses used by the
paper, but it is not checkpoint-compatible with the historical Detectron2
implementation.

## Environment

Use Python 3.10 or newer. Install a CUDA build of PyTorch appropriate for the
machine, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The 2,000-study run was verified on one NVIDIA RTX 3080 with 10 GB of memory.

## Data setup and split generation

Obtain the gated FG-CXR and REFLACX data under their respective access terms.
Extract FG-CXR into the following local layout:

```text
data/fg_cxr/
  full.json
  images/*.jpg
  masks/*.png
  heatmaps/*.png
  heatmaps_reflacx/*.png
```

The annotation release uses REFLACX IDs while its image archive uses DICOM
UUIDs. Generate the local manifest by supplying the gated REFLACX metadata CSV,
which must contain `id` and `dicom_id` columns:

```bash
python -m fgcxr.prepare \
  --reflacx-metadata /path/to/reflacx/metadata.csv
python -m fgcxr.cache_data --workers 8
```

`fgcxr.prepare` validates the local dataset and deterministically generates a
DICOM-group-safe 2,074/295/582 train/validation/test split. No study-ID split
file is distributed. Training selects a deterministic 2,000-study subset from
the generated training split with seed 42.

All dataset files, generated manifests, caches, model checkpoints, predictions,
heatmaps, metrics files, and logs remain under ignored `data/` or `outputs/`
directories and must not be committed.

## Reproduce the study experiment

After data preparation, run the complete training, inference, and evaluation
pipeline with one command:

```bash
bash scripts/reproduce_2k.sh
```

The script performs 6,000 optimizer updates for the gaze predictor, predicts
the seven regional heatmaps, performs 6,000 updates for the report generator,
and evaluates the held-out test set. Set `PYTHON_BIN` to use a specific Python
interpreter. For a fast integration check, run the individual trainers with
`--max-steps 2 --max-val-studies 8`.

## Expected metrics

The paper reports heatmap foreground/background/frequency-weighted IoU of
30.15/89.08/80.69, SSIM 0.600, PSNR 17.41, L1 0.084, and L2 0.022. Its report
generation results are BLEU-1/2/3/4 of 0.729/0.658/0.606/0.561, METEOR 0.386,
ROUGE-L 0.692, CIDEr 4.026, and Div@2 0.854.

Exact values can vary with the generated split, GPU kernels, dependency
versions, and pretrained model revisions. `fgcxr.evaluate` computes the gaze
metrics and all listed language-generation metrics except the paper's CheXbert
clinical-efficacy result, which requires a separately licensed checkpoint.

## IAI attribution and licensing notice

The gaze-segmentation design is based on
[UARK-AICV/IAI](https://github.com/UARK-AICV/IAI) and the WACV 2024 paper
**Decoding Radiologists' Intense Focus for Accurate CXR Diagnoses: A
Controllable & Interpretable AI System** by Trong Thang Pham, Jacob Brecheisen,
Anh Nguyen, Hien Nguyen, and Ngan Le.

The new code under `fgcxr/` is released under the root MIT License. Historical
IAI files under `heatmap_predictor/IAI/` retain their original copyright. The
upstream IAI repository does not currently state a software license, so those
historical files are not relicensed by the root MIT License. Permission from
their copyright holders may be required for reuse or redistribution.

## License

Except for the historical IAI files identified above, the FG-CXR code is
available under the [MIT License](LICENSE).
