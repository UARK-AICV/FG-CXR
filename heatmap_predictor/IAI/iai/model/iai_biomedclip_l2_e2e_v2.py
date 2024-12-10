from typing import List, Any

import open_clip
import torch
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom # what is this big brain function? :)))
from torch import nn
from torch.nn import functional as F

from .clip_utils import (
    FeatureExtractor,
    LearnableBgOvClassifier,
    PredefinedOvClassifier,
    RecWithAttnbiasHead,
    get_predefined_templates,
    ClipText,
)
from .criterion_v2 import SetCriterion_v2
from .matcher import HungarianMatcher
from .side_adapter import build_side_adapter_network_fusetext_l2_v2
@META_ARCH_REGISTRY.register()
class IAIBiomedCLIPL2_e2e_v2(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        clip_visual_extractor: nn.Module,
        clip_rec_head: nn.Module,
        side_adapter_network: nn.Module,
        clip_text: nn.Module,
        ov_classifier: PredefinedOvClassifier,
        criterion: SetCriterion_v2,
        size_divisibility: int,
        asymetric_input: bool = True,
        clip_resolution: float = 0.5,
        pixel_mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
        pixel_std: List[float] = [0.26862954, 0.26130258, 0.27577711],
        sem_seg_postprocess_before_inference: bool = False,
        preprocess_train: Any = None,
        preprocess_val: Any = None,
        tokenizer: Any = None,
    ):
        super().__init__()
        self.asymetric_input = asymetric_input
        self.clip_resolution = clip_resolution
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.size_divisibility = size_divisibility
        self.criterion = criterion

        self.side_adapter_network = side_adapter_network
        self.clip_visual_extractor = clip_visual_extractor
        self.clip_rec_head = clip_rec_head
        self.clip_text = clip_text
        self.ov_classifier = ov_classifier
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        ## copied from maskformer2
        # Loss parameters
        no_object_weight = cfg.MODEL.IAI.NO_OBJECT_WEIGHT
        # loss weights
        class_weight = cfg.MODEL.IAI.CLASS_WEIGHT
        dice_weight = cfg.MODEL.IAI.DICE_WEIGHT
        mask_weight = cfg.MODEL.IAI.MASK_WEIGHT
        try:
            heatmaps_weight = cfg.MODEL.IAI.HEATMAPS_WEIGHT
        except:
            heatmaps_weight = 0.0
        class_weight = cfg.MODEL.IAI.CLASS_WEIGHT
        dice_weight = cfg.MODEL.IAI.DICE_WEIGHT
        mask_weight = cfg.MODEL.IAI.MASK_WEIGHT
        l2_weight = cfg.MODEL.IAI.L2_WEIGHT
        l1_weight = cfg.MODEL.IAI.L1_WEIGHT
        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.IAI.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
            "loss_heatmaps": heatmaps_weight,
            "loss_heatmap_mask": mask_weight,
            "loss_heatmap_dice": dice_weight,
            "loss_l2": l2_weight,
            "loss_l1": l1_weight,

        }
        aux_weight_dict = {}
        for i in range(len(cfg.MODEL.SIDE_ADAPTER.DEEP_SUPERVISION_IDXS) - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        losses = ["labels", "masks", "heatmaps"]

        criterion = SetCriterion_v2(
            num_classes=cfg.MODEL.IAI.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.IAI.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.IAI.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.IAI.IMPORTANCE_SAMPLE_RATIO,
        )
        ## end of copy

        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        preprocess = preprocess_val
        tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        # this one is the text encoder of CLIP. (i didn't know we need text encoder here.)
        # rewrite this forward function. 
        ov_classifier = LearnableBgOvClassifier(
            model, templates=get_predefined_templates(cfg.MODEL.IAI.CLIP_TEMPLATE_SET), tokenizer=tokenizer
        )

        clip_visual_extractor = FeatureExtractor(
            model.visual,
            last_layer_idx=cfg.MODEL.IAI.FEATURE_LAST_LAYER_IDX,
            frozen_exclude=cfg.MODEL.IAI.CLIP_FROZEN_EXCLUDE,
        )
        clip_rec_head = RecWithAttnbiasHead(
            model.visual,
            first_layer_idx=cfg.MODEL.IAI.FEATURE_LAST_LAYER_IDX,
            frozen_exclude=cfg.MODEL.IAI.CLIP_DEEPER_FROZEN_EXCLUDE,
            cross_attn=cfg.MODEL.IAI.REC_CROSS_ATTN,
            sos_token_format=cfg.MODEL.IAI.SOS_TOKEN_FORMAT,
            sos_token_num=cfg.MODEL.SIDE_ADAPTER.NUM_QUERIES,
            downsample_method=cfg.MODEL.IAI.REC_DOWNSAMPLE_METHOD,
        )
        clip_text = ClipText(model.text,
                             tokenizer=tokenizer,
                             frozen_exclude=cfg.MODEL.IAI.CLIP_TEXT_FROZEN_EXCLUDE,
                             )
        pixel_mean, pixel_std = (
            preprocess.transforms[-1].mean,
            preprocess.transforms[-1].std,
        )
        pixel_mean = [255.0 * x for x in pixel_mean]
        pixel_std = [255.0 * x for x in pixel_std]

        return {
            "clip_visual_extractor": clip_visual_extractor,
            "clip_rec_head": clip_rec_head,
            "side_adapter_network": build_side_adapter_network_fusetext_l2_v2(
                cfg, clip_visual_extractor.output_shapes
            ),
            "ov_classifier": ov_classifier,
            "clip_text": clip_text,
            "criterion": criterion,
            "size_divisibility": cfg.MODEL.IAI.SIZE_DIVISIBILITY,
            "asymetric_input": cfg.MODEL.IAI.ASYMETRIC_INPUT,
            "clip_resolution": cfg.MODEL.IAI.CLIP_RESOLUTION,
            "sem_seg_postprocess_before_inference": cfg.MODEL.IAI.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE,
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
            "preprocess_train": preprocess_train,
            "preprocess_val": preprocess_val,
            "tokenizer": tokenizer,

        }

    def forward(self, batched_inputs):
        # get classifier weight for each dataset
        # !! Could be computed once and saved. It will run only once per dataset.
        if "vocabulary" in batched_inputs[0]:
            ov_classifier_weight = (
                self.ov_classifier.logit_scale.exp()
                * self.ov_classifier.get_classifier_by_vocabulary(
                    batched_inputs[0]["vocabulary"]
                )
            )
        else:
            dataset_names = [x["meta"]["dataset_name"] for x in batched_inputs]
            assert (
                len(list(set(dataset_names))) == 1
            ), "All images in a batch must be from the same dataset."
            ov_classifier_weight = (
                self.ov_classifier.logit_scale.exp()
                * self.ov_classifier.get_classifier_by_dataset_name(dataset_names[0])
            )  # C+1,ndim
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        clip_input = images.tensor
        if self.asymetric_input:
            clip_input = F.interpolate(
                clip_input, scale_factor=self.clip_resolution, mode="bilinear"
            )
        clip_image_features = self.clip_visual_extractor(clip_input)
        transcripts = [x["transcript"] for x in batched_inputs]
        clip_text_features = self.clip_text(transcripts, device=clip_input.device)
        
        heatmap_preds, mask_preds, attn_biases = self.side_adapter_network(
            images.tensor, clip_image_features, clip_text_features
        )
        # !! Could be optimized to run in parallel.
        mask_embs = [
            self.clip_rec_head(clip_image_features, attn_bias, normalize=True)
            for attn_bias in attn_biases
        ]  # [B,#queries,C] #queries = cfg.MODEL.SIDE_ADAPTER.NUM_QUERIES
        mask_logits = [
            torch.einsum("bqc,nc->bqn", mask_emb, ov_classifier_weight)
            for mask_emb in mask_embs
        ]


        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            outputs = {
                "pred_logits": mask_logits[-1], # [B, #queries, C], C is the number of classes
                "pred_masks": mask_preds[-1], # [B, #queries, H, W], H,W can be 14, 14
                "pred_heatmaps": heatmap_preds[-1], # [B, #queries, H, W], H,W can be 14, 14
                "aux_outputs": [
                    {
                        "pred_logits": aux_pred_logits,
                        "pred_masks": aux_pred_masks,
                        "pred_heatmaps": aux_pred_heatmaps,
                    }
                    for aux_pred_logits, aux_pred_masks, aux_pred_heatmaps in zip(
                        mask_logits[:-1], mask_preds[:-1], heatmap_preds[:-1]
                    )
                ],
            }
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_preds = mask_preds[-1]
            heatmap_preds = heatmap_preds[-1]
            mask_logits = mask_logits[-1]
            # torch.cuda.empty_cache()
            # Inference
            mask_preds = F.interpolate(
                mask_preds,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=True,
            )
            heatmap_preds = F.interpolate(
                heatmap_preds,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=True,
            )
            processed_results = []
            for mask_cls_result, mask_pred_result, heatmap_pred_result, input_per_image, image_size in zip(
                mask_logits, mask_preds, heatmap_preds, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference: # True
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    heatmap_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        heatmap_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                r1 = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, mask_pred_result
                )
                r2 = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, heatmap_pred_result
                )
                if not self.sem_seg_postprocess_before_inference:
                    r1 = retry_if_cuda_oom(sem_seg_postprocess)(
                        r1, image_size, height, width
                    )
                    r2 = retry_if_cuda_oom(sem_seg_postprocess)(
                        r2, image_size, height, width
                    )
                processed_results[-1]["sem_seg"] = r1
                processed_results[-1]["heatmap"] = r2
            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            gt_heatmaps = targets_per_image.gt_heatmaps
            gt_heatmap_masks = targets_per_image.gt_heatmap_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            padded_heatmaps = torch.zeros(
                (gt_heatmaps.shape[0], h_pad, w_pad),
                dtype=gt_heatmaps.dtype,
                device=gt_heatmaps.device,
            )
            padded_heatmaps[:, : gt_heatmaps.shape[1], : gt_heatmaps.shape[2]] = gt_heatmaps
            padded_heatmap_masks = torch.zeros(
                (gt_heatmap_masks.shape[0], h_pad, w_pad),
                dtype=gt_heatmap_masks.dtype,
                device=gt_heatmap_masks.device,
            )
            padded_heatmap_masks[:, : gt_heatmap_masks.shape[1], : gt_heatmap_masks.shape[2]] = gt_heatmap_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "heatmaps": padded_heatmaps,
                    "heatmap_masks": padded_heatmap_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def train_semantic_inference(self, scale, mask_pred):
        mask_cls = F.softmax(scale, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg
    @property
    def device(self):
        return self.pixel_mean.device
