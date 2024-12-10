from typing import List, Any
import torch
from torch import nn
from torch.nn import functional as F
from open_clip.transformer import VisionTransformer
from detectron2.layers import ShapeSpec
from ..attn_helper import cross_attn_layer, downsample2d, resize_pos_embed2d
import einops


class ClipText(nn.Module):
    def __init__(
        self,
        text_encoder: Any,
        tokenizer: Any,
        frozen_exclude=[],
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.context_length = text_encoder.config.max_position_embeddings
        self.template = 'this is a photo of '
        self._freeze(frozen_exclude)
    def forward(self, text: List[str], attn_mask: torch.Tensor = None, device: str = 'cpu'):
        text = self.tokenizer([self.template + l for l in text], context_length=self.context_length).to(device=device)
        if attn_mask is None:
            attn_mask = (text != self.text_encoder.config.pad_token_id).long()
        out = self.text_encoder.transformer(input_ids=text, attention_mask=attn_mask)
        pooled_out = self.text_encoder.pooler(out, attn_mask)
        projected = self.text_encoder.proj(pooled_out)
        return projected
    def _freeze(self, frozen_exclude=[]):
        for name, param in self.named_parameters():
            if any([x in name for x in frozen_exclude]):
                continue
            param.requires_grad = False