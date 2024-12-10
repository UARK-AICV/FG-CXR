import copy
import logging
import open_clip
import numpy as np
import torch
from torch.nn import functional as F
import SimpleITK as sitk
from PIL import Image
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
from torchvision import transforms
import os
from iai.utils.misc import read_gt_heatmap, read_gt_mask
from PIL import Image

__all__ = ["IAIGazeDatasetMapper_e2e_v2"]


class IAIGazeDatasetMapper_e2e_v2:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        preprocess,
        preprocess_heatmap,
        image_format,
        ignore_label,
        size_divisibility,
        debug_mode,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.preprocess = preprocess
        self.debug_mode = debug_mode

        if isinstance(preprocess.transforms[0].size, int):
            self.input_size = preprocess.transforms[0].size
        else:
            self.input_size = preprocess.transforms[0].size[0] # it is (224, 224), square
        self.preprocess_heatmap = preprocess_heatmap

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info("Current mode: {}".format(mode))





    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        # augs = []
        # augs.append(T.Resize((cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH)))  
        # augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "preprocess": meta.preprocess,
            "preprocess_heatmap": meta.preprocess_heatmap,
            "debug_mode": cfg.DEBUG_MODE,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        dicom_id = dataset_dict["dicom_id"]
        type_ = dataset_dict["heatmap_type"]
        current_dir_this_file  = os.path.dirname(os.path.realpath(__file__))
        last_prefix = f'{dicom_id}_{type_}'

        image_jpg_path = dataset_dict["image_jpg"]
        image = Image.open(image_jpg_path)
        image = self.preprocess(image)
        dataset_dict["image"] = image
        image_shape = (self.input_size, self.input_size)
        instances = Instances(image_shape)
        gt_heatmap_path = dataset_dict["gt_heatmap_path"]
        gt_heatmap_mask, gt_heatmap = read_gt_heatmap(gt_heatmap_path, self.preprocess_heatmap, self.debug_mode)
        gt_mask_path = dataset_dict["gt_mask_path"]
        gt_mask     = read_gt_mask(gt_mask_path, self.preprocess_heatmap)

        classes = np.unique(gt_heatmap_mask)
        classes = classes[classes != self.ignore_label]
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        heatmaps = [] 
        masks = []
        heatmap_masks = []
        for class_id in classes:
            heatmap_masks.append(gt_heatmap_mask == class_id)
            masks.append(gt_mask == class_id)
            if int(class_id) == 0:
                heatmaps.append(1-gt_heatmap) # if we focus on the background.
            else:
                heatmaps.append(gt_heatmap)
        if len(heatmap_masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_heatmap_masks = torch.zeros(
                (0, self.input_size, self.input_size)
            )
            instances.gt_heatmaps = torch.zeros(
                (0, self.input_size, self.input_size)
            )
        else:
            heatmap_masks = BitMasks(
                torch.stack(
                    [
                        torch.from_numpy(np.ascontiguousarray(x.numpy().copy()))
                        for x in heatmap_masks
                    ]
                )
            )
            instances.gt_heatmap_masks = heatmap_masks.tensor
            instances.gt_heatmaps = torch.stack(heatmaps)
        if len(masks) == 0:
            instances.gt_masks = torch.zeros(
                (0, self.input_size, self.input_size)
            )
        else:
            masks = BitMasks(
                torch.stack(
                    [
                        torch.from_numpy(np.ascontiguousarray(x.numpy().copy()))
                        for x in masks
                    ]
                )
            )
            instances.gt_masks = masks.tensor

        dataset_dict["instances"] = instances
        dataset_dict["width"] = self.input_size
        dataset_dict["height"] = self.input_size
        dataset_dict["label"] = torch.tensor(dataset_dict["label"]).long()
        return dataset_dict
