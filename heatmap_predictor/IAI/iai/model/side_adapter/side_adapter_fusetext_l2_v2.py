import logging
from functools import partial
from typing import Dict, List, Tuple, Any

import torch
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.utils.logger import log_first_n
from detectron2.utils.registry import Registry
from timm import create_model
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torch.nn import functional as F

from ..layers import MLP, build_fusion_layer
from .timm_wrapper import PatchEmbed

SIDE_ADAPTER_REGISTRY = Registry("SIDE_ADAPTER")
SIDE_ADAPTER_REGISTRY.__doc__ = """
Registry for side adapter.
"""


def build_side_adapter_network_fusetext_l2_v2(cfg, input_shape):
    name = cfg.MODEL.SIDE_ADAPTER.NAME
    return SIDE_ADAPTER_REGISTRY.get(name)(cfg, input_shape)


class MLPMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        total_heads: int = 1,
        total_layers: int = 1,
        embed_channels: int = 256,
        mlp_channels: int = 256,
        mlp_num_layers: int = 3,
        rescale_attn_bias: bool = False,
    ):
        super().__init__()
        self.total_heads = total_heads
        self.total_layers = total_layers

        dense_affine_func = partial(nn.Conv2d, kernel_size=1)
        # Query Branch: *2 for output dim, one for heatmap, one for mask
        self.query_mlp = MLP(in_channels, mlp_channels, embed_channels*2, mlp_num_layers)
        # Pixel Branch
        self.pix_mlp = MLP(
            in_channels,
            mlp_channels,
            embed_channels,
            mlp_num_layers,
            affine_func=dense_affine_func,
        )
        # Attention Bias Branch
        self.attn_mlp = MLP(
            in_channels,
            mlp_channels,
            embed_channels * self.total_heads * self.total_layers,
            mlp_num_layers,
            affine_func=dense_affine_func,
        )
        if rescale_attn_bias:
            self.bias_scaling = nn.Linear(1, 1)
        else:
            self.bias_scaling = nn.Identity()

    def forward(
        self, query: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # query: [B,N,C]
        # x: [B,C,H,W]
        query_raw = self.query_mlp(query) # [B,N,C*2]
        query_heatmap, query_mask = query_raw.chunk(2, dim=-1)
        pix = self.pix_mlp(x)
        b, c, h, w = pix.shape
        # preidict mask
        heatmap_preds = torch.einsum("bqc,bchw->bqhw", query_heatmap, pix)
        mask_preds = torch.einsum("bqc,bchw->bqhw", query_mask, pix)
        # generate attn bias
        attn = self.attn_mlp(x)
        attn = F.normalize(attn, dim=-1)
        attn = attn.reshape(b, self.total_layers, self.total_heads, c, h, w)
        attn_bias = torch.einsum("bqc,blnchw->blnqhw", query_heatmap, attn)
        attn_bias = self.bias_scaling(attn_bias[..., None]).squeeze(-1)
        attn_bias = attn_bias.chunk(self.total_layers, dim=1)
        attn_bias = [attn.squeeze(1) for attn in attn_bias]
        return heatmap_preds, mask_preds, attn_bias


@SIDE_ADAPTER_REGISTRY.register()
class RegionwiseSideAdapterNetworkFuseTextL2_v2(nn.Module):
    @configurable
    def __init__(
        self,
        vit_model: VisionTransformer,
        fusion_layers: nn.ModuleList,
        mask_decoder: nn.Module,
        text_fusion: nn.Module,
        num_queries: int,
        fusion_map: Dict[int, int],
        deep_supervision_idxs: List[int],
    ):
        super().__init__()
        # remove cls token
        if vit_model.cls_token is not None:
            vit_model.pos_embed = nn.Parameter(vit_model.pos_embed[:, 1:, ...])
        del vit_model.cls_token
        vit_model.cls_token = None
        # delete out norm
        del vit_model.norm
        vit_model.norm = nn.Identity()
        self.vit_model = vit_model
        self.text_fusion = text_fusion

        self.num_queries = num_queries
        self.num_features = vit_model.num_features
        # add query token
        self.query_embed = nn.Parameter(torch.zeros(1, num_queries, self.num_features))
        self.query_pos_embed = nn.Parameter(
            torch.zeros(1, num_queries, self.num_features)
        )
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.normal_(self.query_pos_embed, std=0.02)
        self.fusion_layers = fusion_layers
        self.fusion_map = fusion_map
        self.mask_decoder = mask_decoder
        # for training
        self.deep_supervision_idxs = deep_supervision_idxs

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        vit = create_model(
            cfg.MODEL.SIDE_ADAPTER.VIT_NAME,
            cfg.MODEL.SIDE_ADAPTER.PRETRAINED,
            img_size=cfg.MODEL.SIDE_ADAPTER.IMAGE_SIZE,
            drop_path_rate=cfg.MODEL.SIDE_ADAPTER.DROP_PATH_RATE,
            fc_norm=False,
            num_classes=0,
            embed_layer=PatchEmbed,
        )
        # ["0->0","3->1","6->2","9->3"]
        fusion_map: List[str] = cfg.MODEL.SIDE_ADAPTER.FUSION_MAP

        x2side_map = {int(j): int(i) for i, j in [x.split("->") for x in fusion_map]}
        # build fusion layers
        fusion_type: str = cfg.MODEL.SIDE_ADAPTER.FUSION_TYPE
        fusion_layers = nn.ModuleDict(
            {
                f"layer_{tgt_idx}": build_fusion_layer(
                    fusion_type, input_shape[src_idx].channels, vit.num_features
                )
                for tgt_idx, src_idx in x2side_map.items()
            }
        )
        # NOTE: change 512 to something more dynamic
        text_fusion = build_fusion_layer("add_text", 512, vit.num_features) 
        # build mask decoder
        return {
            "vit_model": vit,
            "num_queries": cfg.MODEL.SIDE_ADAPTER.NUM_QUERIES,
            "fusion_layers": fusion_layers,
            "fusion_map": x2side_map,
            "text_fusion": text_fusion,
            "mask_decoder": MLPMaskDecoder(
                in_channels=vit.num_features,
                total_heads=cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.NUM_HEADS,
                total_layers=cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.NUM_LAYERS,
                embed_channels=cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.EMBED_CHANNELS,
                mlp_channels=cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.MLP_CHANNELS,
                mlp_num_layers=cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.MLP_NUM_LAYERS,
                rescale_attn_bias=cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.RESCALE_ATTN_BIAS,
            ),
            "deep_supervision_idxs": cfg.MODEL.SIDE_ADAPTER.DEEP_SUPERVISION_IDXS,
        }

    def forward(
        self, image: torch.Tensor, clip_features: Dict[Any, torch.Tensor], clip_text_features: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:
        """The main forward function here. 
        
        But it will call forward_features() and decode_masks() to get the final results. 

        """    
        features = self.forward_features(image, clip_features, clip_text_features) # List[Dict[str, torch.Tensor]], keys() = ['query', 'x'], shape = [(bs, #query, dim), (bs, dim, H/?, W/?)], W/? = H/? = 14 in this case
        return self.decode_masks(features) # Tuple[List[torch.Tensor], List[List[torch.Tensor]]], shape = [(bs, #query, H/?, W/?)], [(bs, #query, H/?, W/?)]

    def decode_masks(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        if not self.training:
            features = [features[-1]]
        mask_preds = []
        heatmap_preds = []
        attn_biases = []
        for feature in features:
            heatmap_pred, mask_pred, attn_bias = self.mask_decoder(**feature)
            mask_preds.append(mask_pred)
            heatmap_preds.append(heatmap_pred)
            attn_biases.append(attn_bias)
        return heatmap_preds, mask_preds, attn_biases

    def forward_features(
        self, image: torch.Tensor, clip_features: List[torch.Tensor], clip_text_features: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        # NOTE: if the model performs poorly, try to add raw text with side adapter text encoder like they do for image.
        x, (h, w) = self.vit_model.patch_embed(image) # image shape (bs, 3, H, W), x shape (bs, H/? * W/?, dim), h = w = 14 in this case
        L = x.shape[1]  # token length = H/? * W/? = 196 in this case
        pos_embed = self.vit_model.pos_embed # shape (bs, 1600, dim)
        ori_h, ori_w = self.vit_model.patch_embed.grid_size # ori_h = ori_w = 40 in this case
        if pos_embed.shape[1] != L: # True, so resize pos_embed to (bs, 196, dim), dim is also self.num_features
            pos_embed = (
                F.interpolate(
                    pos_embed.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
                    size=[h, w],
                    mode="bicubic",
                    align_corners=True,
                )
                .flatten(2)
                .permute(0, 2, 1)
            )
        pos_embed = torch.cat(
            [self.query_pos_embed.expand(pos_embed.shape[0], -1, -1), pos_embed], dim=1
        ) # where pos_embed to 296 from 196. Final shape is (bs, 296, dim)
        x = torch.cat(
            [self.query_embed.expand(x.shape[0], -1, -1), x],
            dim=1,
        )  # B, Query+Length, C
        x = x + pos_embed
        x = self.vit_model.norm_pre(x) # shape (bs, 296, dim)
        x = self.fuse(0, x, clip_features, (h, w)) # shape (bs, 296, dim)
        x = self.text_fusion(x, clip_text_features) # shape (bs, 296, dim)
        outs = []
        for i, blk in enumerate(self.vit_model.blocks, start=1): # now looping the remaining blocks in vit_model.blocks
            x = blk(x)
            x = self.fuse(i, x, clip_features, (h, w)) # Add fuse again.
            x = self.text_fusion(x, clip_text_features) # Add text fusion again.
            x = F.normalize(x, dim=-1) # shape (bs, 296, dim) # hack to prevent value explosion.
            if i in self.deep_supervision_idxs: # deep_supervision_idxs = [7, 8] in the config file, if `i` is 7 or 8 (last two blocks of vitmodel.blocks), then append `x` and `query` to `outs
                outs.append(
                    {
                        "query": x[:, :-L, ...],
                        "x": x[:, -L:, ...]
                        .permute(0, 2, 1)
                        .reshape(x.shape[0], x.shape[-1], h, w),
                    }
                )

            if i < len(self.vit_model.blocks): # If not the last block, then add pos_embed to x
                x = x + pos_embed

        return outs

    def fuse(
        self,
        block_idx: int,
        x: torch.Tensor,
        clip_features: List[torch.Tensor],
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        if block_idx in self.fusion_map: # self.fusion_map = {0: 0, 1: 3, 2: 6, 3: 9}
            src_idx = self.fusion_map[block_idx]
            L = spatial_shape[0] * spatial_shape[1] # always hxw =  196, x[:, :-L, ...] shape = (bs, 100, dim), x[:, -L:, ...] shape = (bs, 196, dim)
            x = torch.cat(
                [
                    x[:, :-L, ...],
                    self.fusion_layers[f"layer_{block_idx}"](
                        x[:, -L:, ...], clip_features[src_idx], spatial_shape
                    ),
                ],
                dim=1,
            ) # this step is only encode (or fuse) the last 196 tokens, the concatenate is only to make sure the shape is correct each time. The first 100 dim is query embed. Call it fusion but it is actually just add x[:, -L:, ...] and clip_features[src_idx]. Note that clip_features is reshape from hidden dim to 240 before adding.
            log_first_n(
                logging.INFO,
                f"fuse clip {src_idx} to {block_idx}",
                len(self.fusion_map),
            )
        return x
