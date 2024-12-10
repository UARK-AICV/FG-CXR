from typing import List, Any
import torch
from torch import nn
from torch.nn import functional as F
from open_clip.transformer import VisionTransformer
from detectron2.layers import ShapeSpec
from ..attn_helper import cross_attn_layer, downsample2d, resize_pos_embed2d
import einops

class ClipOutput(dict):
    def __init__(self, spacial_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spacial_shape = spacial_shape

    def save(self, idx: int, clip_feat: torch.Tensor):
        l, n, c = clip_feat.shape
        self[idx] = (
            einops.rearrange(clip_feat[:,1:], "n (h w) c -> n c h w", h=self.spacial_shape[0], w=self.spacial_shape[1])
        )  # n, c, h, w
        self[f"{idx}_cls_token"] = clip_feat[:,0:1,:]  


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        visual_encoder: Any,
        last_layer_idx: int = -1,
        frozen_exclude=[],
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        # self.output_tokens = visual_encoder.output_tokens Timm does not support this, only open_clip does
        self.patch_size = visual_encoder.trunk.patch_embed.patch_size # (16,16) for open_clip
        self.num_features = 768 # 768 for open_clip `    (ln_pre): LayerNorm((768,), eps=1e-05, elementwise_affine=True)`, (norm): Identity() for biomedclip
        if last_layer_idx == -1:
            self.resblocks = visual_encoder.trunk.blocks
            self.last_output_idx = len(self.resblocks) + 1
        else:
            self.resblocks = visual_encoder.trunk.blocks[:last_layer_idx]
            self.last_output_idx = last_layer_idx + 1
        #
        self.frozen_exclude = frozen_exclude
        self._freeze(self.frozen_exclude)

    def forward(self, x):
        x = self.visual_encoder.trunk.patch_embed(x)
        x = self.visual_encoder.trunk._pos_embed(x)
        x = self.visual_encoder.trunk.norm_pre(x)
        h, w = self.visual_encoder.trunk.patch_embed.grid_size
        outputs = ClipOutput(spacial_shape=(h, w))
        outputs.save(0, x)
        for i, resblock in enumerate(self.resblocks, start=1):
            x = resblock(x)
            outputs.save(i, x)
        return outputs

    def _freeze(self, frozen_exclude):
        if "all" in frozen_exclude:
            return
        for name, param in self.named_parameters():
            if not any([exclude in name for exclude in frozen_exclude]):
                param.requires_grad = False

    @property
    def output_shapes(self):
        return {
            i: ShapeSpec(channels=self.num_features)
            for i in range(self.last_output_idx)
        }

    @property
    def size_divisibility(self):
        return self.patch_size[0]


class RecWithAttnbiasHead(nn.Module):
    def __init__(
        self,
        visual_encoder: VisionTransformer,
        first_layer_idx: int = 0,
        frozen_exclude: List[str] = [],
        sos_token_format: str = "cls_token",
        sos_token_num: int = 1,
        cross_attn: bool = True,
        downsample_method: str = "bilinear",
    ):
        super().__init__()
        self.first_layer_idx = first_layer_idx
        self.cross_attn = cross_attn
        self.downsample_method = downsample_method
        self.visual_encoder = visual_encoder

        if first_layer_idx < 0:
            raise NotImplementedError("first_layer_idx < 0 is not implemented yet.")
        self.resblocks = visual_encoder.trunk.blocks[first_layer_idx:]

        self.sos_token_format = sos_token_format
        self.sos_token_num = sos_token_num
        self.frozen_exclude = frozen_exclude

        if sos_token_format in ["learnable_token", "pos_embedding"]:
            self.sos_token = nn.Parameter(
                torch.randn(sos_token_num, 1, self.proj.shape[0])
            )
            nn.init.normal_(self.sos_token, std=0.02)
            self.frozen_exclude.append("sos_token")
        self._freeze(self.frozen_exclude)

    def _freeze(self, frozen_exclude):
        if "all" in frozen_exclude:
            return
        for name, param in self.named_parameters():
            if not any([exclude in name for exclude in frozen_exclude]):
                param.requires_grad = False

    def forward(self, features, attn_bias, normalize: bool = False):
        # construct clip shadow features.
        cls_token = features[f"{self.first_layer_idx}_cls_token"]  
        cls_token = einops.rearrange(cls_token, 'n m c -> m n c') # 1,n,c
        pix_feat = features[self.first_layer_idx]  # n,c,h,w
        n, c, h, w = pix_feat.shape
        x = torch.cat(
            [cls_token, pix_feat.reshape(n, c, -1).permute(2, 0, 1)]
        )  # 1+l,n,c

        # construct sos token.
        if self.sos_token_format == "cls_token":
            sos_token = cls_token.repeat(self.sos_token_num, 1, 1)
        elif self.sos_token_format == "learnable_token":
            sos_token = self.sos_token.expand(-1, n, -1)
        elif self.sos_token_format == "pos_embedding":
            sos_token = self.sos_token.expand(-1, n, -1) + cls_token

        # construct attn biases.
        # this part here will continue the remaining part of the vision transformer.
        # the attention biases are for the cross attention.
        sos_token = einops.rearrange(sos_token, "l n c -> n l c")
        x = einops.rearrange(x, "l n c -> n l c")
        attn_biases = self._build_attn_biases(attn_bias, target_shape=(h, w))
        if self.cross_attn:
            for i, resblock in enumerate(self.resblocks):
                if self.cross_attn:
                    sos_token = cross_attn_layer(
                        resblock,
                        sos_token,
                        x[:,1:,],
                        attn_biases[i],
                    )
                    if i < len(self.resblocks) - 1:
                        x = resblock(x)
        else:
            x = torch.cat([sos_token, x], dim=1)
            for i, resblock in enumerate(self.resblocks):
                x = resblock(x, attn_mask=attn_biases[i])
            sos_token = x[:,: self.sos_token_num,:]
        
        # then permute and normalize as in the end of vision transformer.
        sos_token = self.visual_encoder.trunk.norm(sos_token)
        sos_token = self.visual_encoder.trunk.forward_head(sos_token, is_iai=True)
        # this one is the real head that reduce to 512
        sos_token = self.visual_encoder.head(sos_token)
        if normalize:
            sos_token = F.normalize(sos_token, dim=-1)
        return sos_token

    def _build_attn_biases(self, attn_biases, target_shape):
        formatted_attn_biases = []
        for attn_bias in attn_biases:
            # convert it to proper format: N*num_head,L,L
            # attn_bias: [N, num_head/1, num_sos,H,W]
            n, num_head, num_sos, h, w = attn_bias.shape
            # reshape and downsample
            attn_bias = downsample2d(
                attn_bias.reshape(n, num_head * num_sos, h, w),
                target_shape,
                method=self.downsample_method,
            )
            attn_bias = attn_bias.reshape(n, num_head, num_sos, *target_shape)
            true_num_head = self.resblocks[0].attn.num_heads
            assert (
                num_head == 1 or num_head == true_num_head
            ), f"num_head={num_head} is not supported."
            if num_head == 1:
                attn_bias = attn_bias.repeat(1, true_num_head, 1, 1, 1)
            attn_bias = attn_bias.reshape(n * true_num_head, num_sos, -1)
            L = attn_bias.shape[-1]
            if self.cross_attn:
                # [n*num_head, num_sos, L]
                formatted_attn_biases.append(attn_bias)
            else:
                # [n*num_head, num_sos+1+L, num_sos+1+L]
                new_attn_bias = attn_bias.new_zeros(num_sos + 1 + L, num_sos + 1 + L)
                new_attn_bias[:, :num_sos] = -100
                new_attn_bias[torch.arange(num_sos), torch.arange(num_sos)] = 0
                new_attn_bias[:num_sos, num_sos] = -100
                new_attn_bias = (
                    new_attn_bias[None, ...].expand(n * true_num_head, -1, -1).clone()
                )
                new_attn_bias[..., :num_sos, -L:] = attn_bias
                formatted_attn_biases.append(new_attn_bias)

        if len(formatted_attn_biases) == 1:
            formatted_attn_biases = [formatted_attn_biases[0] for _ in self.resblocks]
        return formatted_attn_biases
