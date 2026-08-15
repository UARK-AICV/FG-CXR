from __future__ import annotations

import open_clip
import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, CvtModel, GPT2LMHeadModel

from ..constants import BIOMEDCLIP_MODEL, CVT_MODEL, DECODER_MODEL, REGIONS


class FGReportGenerator(nn.Module):
    def __init__(
        self,
        encoder: CvtModel,
        decoder: GPT2LMHeadModel,
        anatomy_text_features: torch.Tensor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_projection = nn.Linear(384, decoder.config.n_embd)
        self.register_buffer("anatomy_text_features", anatomy_text_features.float(), persistent=True)
        self.direction_projection = nn.Linear(anatomy_text_features.shape[-1], decoder.config.n_embd)
        self.decoder = decoder

    def visual_context(self, images: torch.Tensor, heatmaps: torch.Tensor) -> torch.Tensor:
        features = self.encoder(pixel_values=images, return_dict=True).last_hidden_state
        features = features.flatten(2).transpose(1, 2)
        features = self.encoder_projection(features)
        batch, regions, height, width = heatmaps.shape
        if features.shape[1] != height * width:
            raise ValueError(f"CvT emitted {features.shape[1]} patches but heatmaps have {height}x{width}")
        masked = features[:, None] * heatmaps.flatten(2)[..., None]
        direction = self.direction_projection(self.anatomy_text_features)
        direction = direction[None, :, None].expand(batch, -1, -1, -1)
        return torch.cat((masked, direction), dim=2).flatten(0, 1)

    def forward(
        self,
        images: torch.Tensor,
        heatmaps: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        context = self.visual_context(images, heatmaps)
        output = self.decoder(
            input_ids=input_ids.flatten(0, 1),
            attention_mask=attention_mask.flatten(0, 1),
            labels=labels.flatten(0, 1),
            encoder_hidden_states=context,
            encoder_attention_mask=torch.ones(context.shape[:2], dtype=torch.long, device=context.device),
            return_dict=True,
        )
        return output.loss

    @torch.inference_mode()
    def generate(
        self,
        images: torch.Tensor,
        heatmaps: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        max_new_tokens: int = 60,
        num_beams: int = 1,
    ) -> torch.Tensor:
        context = self.visual_context(images, heatmaps)
        input_ids = torch.full(
            (context.shape[0], 1), bos_token_id, dtype=torch.long, device=context.device
        )
        return self.decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=context,
            encoder_attention_mask=torch.ones(context.shape[:2], dtype=torch.long, device=context.device),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=True,
        )


def build_tokenizer(model_name: str = DECODER_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.add_special_tokens({"bos_token": "[BOS]", "pad_token": "[PAD]"})
    return tokenizer


def build_report_model(
    cvt_model: str = CVT_MODEL,
    decoder_model: str = DECODER_MODEL,
    biomedclip_model: str = BIOMEDCLIP_MODEL,
) -> tuple[FGReportGenerator, object]:
    tokenizer = build_tokenizer(decoder_model)
    config = AutoConfig.from_pretrained(decoder_model)
    config.add_cross_attention = True
    config.is_decoder = True
    config.use_cache = False
    decoder = GPT2LMHeadModel.from_pretrained(decoder_model, config=config)
    # Mean/covariance initialization is important for GPT-2: randomly centered
    # new embeddings otherwise dominate the pretrained vocabulary logits and
    # make the initial language-model loss jump from ~8 to >40.
    decoder.resize_token_embeddings(len(tokenizer))
    encoder = CvtModel.from_pretrained(cvt_model)

    clip_model, _, _ = open_clip.create_model_and_transforms(biomedclip_model)
    clip_tokenizer = open_clip.get_tokenizer(biomedclip_model)
    clip_model.eval()
    with torch.inference_mode():
        text_features = clip_model.encode_text(clip_tokenizer(list(REGIONS)), normalize=True).cpu()
    text_features = text_features.clone()
    del clip_model
    return FGReportGenerator(encoder, decoder, text_features), tokenizer
