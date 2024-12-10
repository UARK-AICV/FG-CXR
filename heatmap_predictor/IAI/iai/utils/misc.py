# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""
from typing import List, Optional
from functools import reduce
import torch
import torch.distributed as dist
import torchvision
from torch import Tensor
from torch.nn import functional as F
import SimpleITK as sitk
from PIL import Image
import os 
import numpy as np

def read_dicoms(image_path):
    image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image)
    image = image/np.max(image)
    image = np.moveaxis(image, 0, -1)
    image = image.repeat(3, axis=-1)
    image = Image.fromarray((image * 255).astype(np.uint8))
    return image

def read_gt_heatmap(heatmap_path, preprocess_heatmap, debug_mode=0):
    if debug_mode==1:
        reflacx_id = heatmap_path.split('/')[-2]
        if reflacx_id[0] != 'P':
            reflacx_id = heatmap_path.split('/')[-1].split('.')[0]
        saved_dir = 'tmps'
        os.makedirs(saved_dir, exist_ok=True)
    if '/home/tp030/SAMed' in heatmap_path:
        heatmap_path = heatmap_path.replace('/home/tp030/SAMed', '/home/tp030/IAI')
    gt_heatmap = Image.open(heatmap_path).convert('L') # i convert it so that i can transforms.
    gt_heatmap = preprocess_heatmap(gt_heatmap)
    gt_heatmap_bf = gt_heatmap.clone() # the raw logits
    gt_heatmap_bf = gt_heatmap_bf[0]
    # gt_thresdhold = gt_heatmap.unique()[1]
    gt_thresdhold = gt_heatmap.unique()[len(np.unique(gt_heatmap))//5]
    gt_heatmap[gt_heatmap > gt_thresdhold] = 1 # class count from 1
    gt_heatmap[gt_heatmap <= gt_thresdhold] = 0 # outside class
    # make sure it is a mask HxW with no channel dim whatsoever
    gt_heatmap = gt_heatmap[0].long()
    if debug_mode==1:
        gt_heatmap_s = gt_heatmap.clone().squeeze(0).numpy()
        gt_heatmap_s = Image.fromarray((gt_heatmap_s*255).astype(np.uint8)) # *255 but if you have more than one class, i think you need to do something else.
        gt_heatmap_s.save(f'{saved_dir}/{reflacx_id}_gt_heatmap.jpg')
        gt_heatmap_bf_s = gt_heatmap_bf.clone().squeeze(0).numpy()
        gt_heatmap_bf_s = Image.fromarray((gt_heatmap_bf_s*255).astype(np.uint8)) # *255 but if you have more than one class, i think you need to do something else.
        gt_heatmap_bf_s.save(f'{saved_dir}/{reflacx_id}_gt_heatmap_bf.jpg')
    # exit()
    return gt_heatmap, gt_heatmap_bf

def read_gt_mask(mask_path, preprocess):
    c = Image.open(mask_path).convert('L')
    c = preprocess(c)
    c = c>40/255
    c = c[0].long()
    return c

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(
            img, (0, padding[2], 0, padding[1], 0, padding[0])
        )
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(
            m, (0, padding[2], 0, padding[1]), "constant", 1
        )
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_module_by_name(module, access_string):
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)
