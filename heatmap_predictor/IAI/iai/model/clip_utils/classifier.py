from typing import List
from torch.nn import functional as F
import torch
from detectron2.utils.registry import Registry
from open_clip.model import CLIP
from torch import nn
from .utils import get_labelset_from_dataset
from open_clip import tokenizer
from open_clip.hf_model import ClsPooler


class PredefinedOvClassifier(nn.Module):
    def __init__(
        self,
        clip_model: CLIP,
        cache_feature: bool = True,
        templates: List[str] = ["a photo of {}"],
        tokenizer=tokenizer,
    ):
        # copy the clip model to this module
        super().__init__()
        for name, child in clip_model.named_children():
            if "visual" not in name:
                self.add_module(name, child)
        for name, param in clip_model.named_parameters(recurse=False):
            self.register_parameter(name, param)
        for name, buffer in clip_model.named_buffers(recurse=False):
            self.register_buffer(name, buffer)
        self.templates = templates
        self._freeze()
        self.tokenizer = tokenizer
        self.cache_feature = cache_feature
        self.clip_model_text = clip_model.text
        if self.cache_feature:
            self.cache = {}

    def forward(self, category_names: List[str]):
        text_embed_bucket = []
        for template in self.templates:
            noun_tokens = self.tokenizer(
                [template.format(noun) for noun in category_names]
            )
            text_inputs = noun_tokens.to(self.text.proj[0].weight.data.device)
            text_embed = self.encode_text(text_inputs, normalize=True)
            text_embed_bucket.append(text_embed)
        text_embed = torch.stack(text_embed_bucket).mean(dim=0)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        return text_embed

    @torch.no_grad()
    def encode_text(self, text, normalize: bool = False):
        attn_mask = (text != self.clip_model_text.config.pad_token_id).long()
        out = self.clip_model_text.transformer(input_ids=text, attention_mask=attn_mask)
        pooled_out = self.clip_model_text.pooler(out, attn_mask)
        projected = self.clip_model_text.proj(pooled_out)

        seq_len = out.last_hidden_state.shape[1]
        tokens = (
            out.last_hidden_state[:, torch.arange(seq_len) != self.clip_model_text.pooler.cls_token_position, :] 
            if type(self.clip_model_text.pooler) == ClsPooler 
            else out.last_hidden_state
        )
        
        if self.clip_model_text.output_tokens:
            return projected, tokens
        # return projected
        return F.normalize(projected, dim=-1) if normalize else projected

    def get_classifier_by_vocabulary(self, vocabulary: List[str]):
        if self.cache_feature:
            new_words = [word for word in vocabulary if word not in self.cache]
            if len(new_words) > 0:
                cat_embeddings = self(new_words)
                self.cache.update(dict(zip(new_words, cat_embeddings)))
            cat_embeddings = torch.stack([self.cache[word] for word in vocabulary])
        else:
            cat_embeddings = self(vocabulary)
        return cat_embeddings

    def get_classifier_by_dataset_name(self, dataset_name: str):
        if self.cache_feature:
            if dataset_name not in self.cache:
                category_names = get_labelset_from_dataset(dataset_name)
                cat_embeddings = self(category_names)
                self.cache[dataset_name] = cat_embeddings
            cat_embeddings = self.cache[dataset_name]
        else:
            category_names = get_labelset_from_dataset(dataset_name)
            cat_embeddings = self(category_names)
        return cat_embeddings

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(False)


class LearnableBgOvClassifier(PredefinedOvClassifier):
    def __init__(
        self,
        clip_model: CLIP,
        cache_feature: bool = True,
        templates: List[str] = ["a photo of {}"],
        tokenizer=tokenizer,
    ):
        super().__init__(clip_model, cache_feature, templates, tokenizer=tokenizer)
        self.bg_embed = nn.Parameter(torch.randn(1, self.text.proj[-1].weight.shape[0]))
        nn.init.normal_(
            self.bg_embed,
            std=self.bg_embed.shape[1] ** -0.5,
        )

    def get_classifier_by_vocabulary(self, vocabulary: List[str]):
        cat_embedding = super().get_classifier_by_vocabulary(vocabulary)
        cat_embedding = torch.cat([cat_embedding, self.bg_embed], dim=0)
        cat_embedding = F.normalize(cat_embedding, p=2, dim=-1)
        return cat_embedding

    def get_classifier_by_dataset_name(self, dataset_name: str):
        cat_embedding = super().get_classifier_by_dataset_name(dataset_name)
        cat_embedding = torch.cat([cat_embedding, self.bg_embed], dim=0)
        cat_embedding = F.normalize(cat_embedding, p=2, dim=-1)
        return cat_embedding
