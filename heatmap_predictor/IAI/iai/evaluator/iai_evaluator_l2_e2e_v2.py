# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import DatasetEvaluator
from iai.utils.misc import  read_gt_heatmap, read_gt_mask

from torch.nn import functional as F
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import torch


_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False

class IAIEvaluatorL2_e2e_v2(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        cfg=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "IAIEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "IAIEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")


        # We have two inputs, not one. We need a way to merge it into one key.
        self.make_key = lambda x, y: x + "|" + y
        self.input_file_to_gt_file = {
            self.make_key(dataset_record["image_jpg"], dataset_record["transcript"]): (dataset_record["gt_heatmap_path"], dataset_record["gt_mask_path"])
            for dataset_record in DatasetCatalog.get(dataset_name)
        }


        meta = MetadataCatalog.get(dataset_name)
        self.preprocess_heatmap = meta.preprocess_heatmap
        self.preprocess = meta.preprocess
        self.debug_mode = cfg.DEBUG_MODE
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.thing_classes
        self._num_classes = len(meta.thing_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        # This is because cv2.erode did not work for int datatype. Only works for uint8.
        self._compute_boundary_iou = True
        if not _CV2_IMPORTED:
            self._compute_boundary_iou = False
            self._logger.warn(
                """Boundary IoU calculation requires OpenCV. B-IoU metrics are
                not going to be computed because OpenCV is not available to import."""
            )
        if self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False
            self._logger.warn(
                f"""IAIEvaluator(num_classes) is more than supported value for Boundary IoU calculation!
                B-IoU metrics are not going to be computed. Max allowed value (exclusive)
                for num_classes for calculating Boundary IoU is {np.iinfo(np.uint8).max}.
                The number of classes of dataset {self._dataset_name} is {self._num_classes}"""
            )
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._b_conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self._conf_matrix_heatmap = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._b_conf_matrix_heatmap  = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self._predictions = []
        self._custom_metrics = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):

            output_raw_mask = output["sem_seg"].to(self._cpu_device) #[2, 224, 224]
            output_raw_heatmap = output["heatmap"].to(self._cpu_device) # [1, 224, 224]
            
            # scale /max for each dim
            output_raw_mask[1] = output_raw_mask[1] / output_raw_mask[1].max()
            output_raw_heatmap[1] = output_raw_heatmap[1] / output_raw_heatmap[1].max()

            output_mask = output["sem_seg"].argmax(dim=0).to(self._cpu_device) # this right here is picking one class. So there is no hope for you doing multiclass classification.
            output_heatmap = output["heatmap"].argmax(dim=0).to(self._cpu_device) # [1, 224, 224]
            pred_mask = np.array(output_mask, dtype=np.int)
            pred_heatmap = np.array(output_heatmap, dtype=np.int)
            # except this part, the remaining should be ok.
            gt_filename, gt_mask_path = self.input_file_to_gt_file[self.make_key(input["image_jpg"], input['transcript'])]
            gt_heatmap, gt_heatmap_bf = read_gt_heatmap(gt_filename, self.preprocess_heatmap, self.debug_mode)
            gt_mask = read_gt_mask(gt_mask_path, self.preprocess_heatmap)

            gt_heatmap = np.array(gt_heatmap).astype(int) # [224, 224]
            gt_mask = np.array(gt_mask).astype(int) # [224, 224]

            gt_heatmap[gt_heatmap == self._ignore_label] = self._num_classes
            gt_mask[gt_mask == self._ignore_label] = self._num_classes

            # horizontal is gt, vertical is pred
            self._conf_matrix_heatmap += np.bincount(
                (self._num_classes + 1) * pred_heatmap.reshape(-1) + gt_heatmap.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)
            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred_mask.reshape(-1) + gt_mask.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt_heatmap.astype(np.uint8))
                b_pred_heatmap = self._mask_to_boundary(pred_heatmap.astype(np.uint8))
                self._b_conf_matrix_heatmap += np.bincount(
                    (self._num_classes + 1) * b_pred_heatmap.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)
                b_gt = self._mask_to_boundary(gt_mask.astype(np.uint8))
                b_pred_mask = self._mask_to_boundary(pred_mask.astype(np.uint8))
                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred_mask.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(b_pred_mask, input["image_jpg"], input['transcript'], input['heatmap_type']))

            if gt_heatmap_bf.unique().shape[0] == 1:
                gt_heatmap_bf = gt_heatmap_bf * 0
            else:
                gt_heatmap_bf = gt_heatmap_bf / torch.max(gt_heatmap_bf)

            output_mask = output_mask.float()
            metrics = {"l2_loss": F.mse_loss(output_raw_heatmap[-1], gt_heatmap_bf).item(),
                       "l1_loss": F.l1_loss(output_raw_heatmap[-1], gt_heatmap_bf).item(),
                       "ssim": self.ssim(output_raw_heatmap[-1].view(1,1,224,224), gt_heatmap_bf.view(1,1,224,224)).item(),
                       "ssim_mask": self.ssim(output_mask.view(1,1,224,224), gt_heatmap_bf.view(1,1,224,224)).item(),
                        "psnr": self.psnr(output_raw_heatmap[-1], gt_heatmap_bf).item(),
                        "jpg": input["image_jpg"],
            }
            self._custom_metrics.append(metrics)

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            b_conf_matrix_list = all_gather(self._b_conf_matrix)
            conf_matrix_heatmap_list = all_gather(self._conf_matrix_heatmap)
            b_conf_matrix_heatmap_list = all_gather(self._b_conf_matrix_heatmap)
            self._predictions = all_gather(self._predictions)
            self._custom_metrics = all_gather(self._custom_metrics)
            self._custom_metrics = list(itertools.chain(*self._custom_metrics))
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix
            self._conf_matrix_heatmap = np.zeros_like(self._conf_matrix_heatmap)
            for conf_matrix_heatmap in conf_matrix_heatmap_list:
                self._conf_matrix_heatmap += conf_matrix_heatmap


            self._b_conf_matrix = np.zeros_like(self._b_conf_matrix)
            for b_conf_matrix in b_conf_matrix_list:
                self._b_conf_matrix += b_conf_matrix
            self._b_conf_matrix_heatmap = np.zeros_like(self._b_conf_matrix_heatmap)
            for b_conf_matrix_heatmap in b_conf_matrix_heatmap_list:
                self._b_conf_matrix_heatmap += b_conf_matrix_heatmap

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))
            file_path_classification = os.path.join(self._output_dir, "classification_predictions.json")
            with PathManager.open(file_path_classification, "w") as f:
                f.write(json.dumps(self._custom_metrics))

        # mask side
        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)
        fgiou = iou[1]
        bgiou = iou[0]
        if self._compute_boundary_iou:
            b_iou = np.full(self._num_classes, np.nan, dtype=np.float)
            b_tp = self._b_conf_matrix.diagonal()[:-1].astype(np.float)
            b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(np.float)
            b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(np.float)
            b_union = b_pos_gt + b_pos_pred - b_tp
            b_iou_valid = b_union > 0
            b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            if self._compute_boundary_iou:
                res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]
                res[f"min(IoU, B-Iou)-{name}"] = 100 * min(iou[i], b_iou[i])
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}"] = 100 * acc[i]
        res["mSSIM_mask"] = np.mean([x['ssim_mask'] for x in self._custom_metrics])
        res["fgIoU"] = 100 * fgiou
        res["bgIoU"] = 100 * bgiou


        # heatmap side
        acc_heatmap = np.full(self._num_classes, np.nan, dtype=np.float)
        iou_heatmap = np.full(self._num_classes, np.nan, dtype=np.float)
        tp_heatmap = self._conf_matrix_heatmap.diagonal()[:-1].astype(np.float)
        pos_gt_heatmap = np.sum(self._conf_matrix_heatmap[:-1, :-1], axis=0).astype(np.float)
        class_weights_heatmap = pos_gt_heatmap / np.sum(pos_gt_heatmap)
        pos_pred_heatmap = np.sum(self._conf_matrix_heatmap[:-1, :-1], axis=1).astype(np.float)
        acc_valid_heatmap = pos_gt_heatmap > 0
        acc_heatmap[acc_valid_heatmap] = tp_heatmap[acc_valid_heatmap] / pos_gt_heatmap[acc_valid_heatmap]
        union_heatmap = pos_gt_heatmap + pos_pred_heatmap - tp_heatmap
        iou_valid_heatmap = np.logical_and(acc_valid_heatmap, union_heatmap > 0)
        iou_heatmap[iou_valid_heatmap] = tp_heatmap[iou_valid_heatmap] / union_heatmap[iou_valid_heatmap]
        macc_heatmap = np.sum(acc_heatmap[acc_valid_heatmap]) / np.sum(acc_valid_heatmap)
        miou_heatmap = np.sum(iou_heatmap[iou_valid_heatmap]) / np.sum(iou_valid_heatmap)
        fiou_heatmap = np.sum(iou_heatmap[iou_valid_heatmap] * class_weights_heatmap[iou_valid_heatmap])
        pacc_heatmap = np.sum(tp_heatmap) / np.sum(pos_gt_heatmap)
        fgiou_heatmap = iou_heatmap[1]
        bgiou_heatmap = iou_heatmap[0]
        if self._compute_boundary_iou:
            b_iou_heatmap = np.full(self._num_classes, np.nan, dtype=np.float)
            b_tp_heatmap = self._b_conf_matrix_heatmap.diagonal()[:-1].astype(np.float)
            b_pos_gt_heatmap = np.sum(self._b_conf_matrix_heatmap[:-1, :-1], axis=0).astype(np.float)
            b_pos_pred_heatmap = np.sum(self._b_conf_matrix_heatmap[:-1, :-1], axis=1).astype(np.float)
            b_union_heatmap = b_pos_gt_heatmap + b_pos_pred_heatmap - b_tp_heatmap
            b_iou_valid_heatmap = b_union_heatmap > 0
            b_iou_heatmap[b_iou_valid_heatmap] = b_tp_heatmap[b_iou_valid_heatmap] / b_union_heatmap[b_iou_valid_heatmap]

        res["mIoU_heatmap"] = 100 * miou_heatmap
        res["fwIoU_heatmap"] = 100 * fiou_heatmap
        for i, name in enumerate(self._class_names):
            res[f"IoU_heatmap-{name}"] = 100 * iou_heatmap[i]
            if self._compute_boundary_iou:
                res[f"BoundaryIoU_heatmap-{name}"] = 100 * b_iou_heatmap[i]
                res[f"min(IoU, B-Iou)_heatmap-{name}"] = 100 * min(iou_heatmap[i], b_iou_heatmap[i])
        res["mACC_heatmap"] = 100 * macc_heatmap
        res["pACC_heatmap"] = 100 * pacc_heatmap
        res["mSSIM"] = np.mean([x['ssim'] for x in self._custom_metrics])
        res["mPSNR"] = np.mean([x['psnr'] for x in self._custom_metrics])
        res["mL1"] = np.mean([x['l1_loss'] for x in self._custom_metrics])
        res["mL2"] = np.mean([x['l2_loss'] for x in self._custom_metrics])
        res["fgIoU_heatmap"] = 100 * fgiou_heatmap
        res["bgIoU_heatmap"] = 100 * bgiou_heatmap
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name, transcript="", heatmap_type="Cardiomegaly"):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "heatmap_type": heatmap_type, "transcript":transcript, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list

    def _mask_to_boundary(self, mask: np.ndarray, dilation_ratio=0.02):
        assert mask.ndim == 2, "mask_to_boundary expects a 2-dimensional image"
        h, w = mask.shape
        diag_len = np.sqrt(h**2 + w**2)
        dilation = max(1, int(round(dilation_ratio * diag_len)))
        kernel = np.ones((3, 3), dtype=np.uint8)

        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        eroded_mask_with_padding = cv2.erode(padded_mask, kernel, iterations=dilation)
        eroded_mask = eroded_mask_with_padding[1:-1, 1:-1]
        boundary = mask - eroded_mask
        return boundary
