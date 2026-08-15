from __future__ import annotations

from dataclasses import dataclass

import open_clip
import torch
from torch import nn
from torch.nn import functional as F

from ..constants import BIOMEDCLIP_MODEL, REGIONS


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int = 3) -> None:
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * (layers - 1) + [output_dim]
        modules: list[nn.Module] = []
        for index in range(len(dimensions) - 1):
            modules.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index != len(dimensions) - 2:
                modules.append(nn.GELU())
        self.network = nn.Sequential(*modules)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class IAIGazePredictor(nn.Module):
    """A dependency-light reconstruction of the paper's IAI side adapter.

    It keeps the recovered experiment's BiomedCLIP layers (0, 3, 6, 9),
    240-dimensional/6-head/8-layer side ViT, anatomy text conditioning, and
    three-layer 256-dimensional mask decoder. It predicts the seven fixed
    anatomical regions together, avoiding Detectron2's 100-query Hungarian
    wrapper without changing the region-conditioned heatmap objective.
    """

    def __init__(
        self,
        visual: nn.Module,
        anatomy_text_features: torch.Tensor,
        hidden_dim: int = 240,
        heads: int = 6,
        depth: int = 8,
        mlp_dim: int = 256,
    ) -> None:
        super().__init__()
        self.visual = visual
        for parameter in self.visual.parameters():
            parameter.requires_grad = False
        self.visual.eval()

        self.clip_feature_indices = (0, 3, 6, 9)
        clip_dim = 768
        grid_size = tuple(self.visual.trunk.patch_embed.grid_size)
        if grid_size != (14, 14):
            raise ValueError(f"Expected BiomedCLIP 14x14 patch grid, got {grid_size}")

        self.side_patch = nn.Conv2d(3, hidden_dim, kernel_size=16, stride=16)
        self.patch_position = nn.Parameter(torch.zeros(1, grid_size[0] * grid_size[1], hidden_dim))
        self.text_projection = nn.Linear(anatomy_text_features.shape[-1], hidden_dim)
        self.register_buffer("anatomy_text_features", anatomy_text_features.float(), persistent=True)
        self.clip_fusions = nn.ModuleList(nn.Linear(clip_dim, hidden_dim) for _ in range(4))
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.query_mlp = MLP(hidden_dim, mlp_dim, mlp_dim, layers=3)
        self.pixel_mlp = MLP(hidden_dim, mlp_dim, mlp_dim, layers=3)
        nn.init.trunc_normal_(self.patch_position, std=0.02)

    def train(self, mode: bool = True) -> "IAIGazePredictor":
        super().train(mode)
        self.visual.eval()
        return self

    @torch.no_grad()
    def _clip_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        trunk = self.visual.trunk
        value = trunk.patch_embed(images)
        value = trunk._pos_embed(value)
        value = trunk.norm_pre(value)
        outputs = [value[:, 1:]]
        for block_index, block in enumerate(trunk.blocks, start=1):
            value = block(value)
            if block_index in self.clip_feature_indices[1:]:
                outputs.append(value[:, 1:])
            if block_index == self.clip_feature_indices[-1]:
                break
        if len(outputs) != 4:
            raise RuntimeError(f"Expected four BiomedCLIP feature maps, got {len(outputs)}")
        return outputs

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        clip_features = self._clip_features(images)
        patch = self.side_patch(images).flatten(2).transpose(1, 2) + self.patch_position
        query = self.text_projection(self.anatomy_text_features).unsqueeze(0).expand(images.shape[0], -1, -1)
        value = torch.cat((query, patch), dim=1)
        region_count = len(REGIONS)
        auxiliary: list[torch.Tensor] = []
        for block_index, block in enumerate(self.blocks):
            if block_index < len(self.clip_fusions):
                value[:, region_count:] = value[:, region_count:] + self.clip_fusions[block_index](clip_features[block_index])
            value = block(value)
            if block_index >= len(self.blocks) - 2:
                normalized = self.final_norm(value)
                query_features = self.query_mlp(normalized[:, :region_count])
                pixel_features = self.pixel_mlp(normalized[:, region_count:])
                auxiliary.append(torch.einsum("brc,bpc->brp", query_features, pixel_features).reshape(-1, region_count, 14, 14))
        return {"logits": auxiliary[-1], "aux_logits": auxiliary[:-1]}


@dataclass
class HeatmapLoss:
    total: torch.Tensor
    l2: torch.Tensor
    soft_bce: torch.Tensor
    mask_bce: torch.Tensor
    dice: torch.Tensor


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = logits.sigmoid().flatten(2)
    targets = targets.flatten(2)
    numerator = 2 * (predictions * targets).sum(-1) + 1
    denominator = predictions.sum(-1) + targets.sum(-1) + 1
    return (1 - numerator / denominator).mean()


def heatmap_loss(outputs: dict[str, torch.Tensor | list[torch.Tensor]], heatmaps: torch.Tensor, masks: torch.Tensor) -> HeatmapLoss:
    logits = outputs["logits"]
    assert isinstance(logits, torch.Tensor)
    resized_heatmaps = F.interpolate(heatmaps, size=logits.shape[-2:], mode="bilinear", align_corners=False)
    resized_masks = F.interpolate(masks, size=logits.shape[-2:], mode="nearest")
    target_logits = torch.logit(resized_heatmaps.clamp(0.01, 0.99))
    l2 = F.mse_loss(logits, target_logits)
    soft_bce = F.binary_cross_entropy_with_logits(logits, resized_heatmaps)
    mask_bce = F.binary_cross_entropy_with_logits(logits, resized_masks)
    dice = _dice_loss(logits, resized_masks)
    total = l2 + soft_bce + mask_bce + dice
    aux_logits = outputs["aux_logits"]
    assert isinstance(aux_logits, list)
    for aux in aux_logits:
        total = total + 0.5 * (
            F.mse_loss(aux, target_logits)
            + F.binary_cross_entropy_with_logits(aux, resized_heatmaps)
            + F.binary_cross_entropy_with_logits(aux, resized_masks)
            + _dice_loss(aux, resized_masks)
        )
    return HeatmapLoss(total=total, l2=l2, soft_bce=soft_bce, mask_bce=mask_bce, dice=dice)


def build_iai_model(model_name: str = BIOMEDCLIP_MODEL) -> IAIGazePredictor:
    clip_model, _, _ = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    clip_model.eval()
    with torch.inference_mode():
        text_features = clip_model.encode_text(tokenizer(list(REGIONS)), normalize=True).cpu()
    # Tensors created in inference mode reject state_dict's in-place copy.
    text_features = text_features.clone()
    return IAIGazePredictor(clip_model.visual, text_features)
