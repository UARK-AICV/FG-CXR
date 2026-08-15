from __future__ import annotations

REGIONS = (
    "heart",
    "left",
    "upper left",
    "lower left",
    "right",
    "upper right",
    "lower right",
)

REGION_TO_INDEX = {name: index for index, name in enumerate(REGIONS)}

BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CVT_MODEL = "microsoft/cvt-21-384-22k"
DECODER_MODEL = "distilbert/distilgpt2"

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
