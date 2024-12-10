import numpy as np
from tools.dataset.dataset import Subset
import re
import torch
import cv2 
class TaskSubset(Subset):
    """
    A subset of the task's dataset. Implemented using a torch.utils.data.Dataset
    for the torch.utils.data.DataLoader for the subset of the
    pytorch_lightning.core.datamodule.LightningDataModule. See the tutorial
    here: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __getitem__(self, index):
        example = self.examples[index]
        image = self.image_loading_and_preprocessing(example["image_file_path"][0])
        example_dict = {
            "id": example["id"],
            "encoder_images": image,
            "direction": example["direction"],
            "labels": example["label"],
            'image_filepaths': example["image_file_path"][0],
        }
        if self.train and not self.self_critical:
            example_dict = {**example_dict, **self.tokenize(example["label"])}
        return example_dict


class TaskSubsetHeatmap(Subset):
    """
    A subset of the task's dataset. Implemented using a torch.utils.data.Dataset
    for the torch.utils.data.DataLoader for the subset of the
    pytorch_lightning.core.datamodule.LightningDataModule. See the tutorial
    here: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __getitem__(self, index):
        example = self.examples[index]
        image = self.image_loading_and_preprocessing(example["image_file_path"][0])
        heatmap = cv2.imread(example["heatmap_file_path"][0], cv2.IMREAD_GRAYSCALE)
        # shape of encoded feature. 
        # TODO: make this dynamic follow the model's output shape
        heatmap = cv2.resize(heatmap, (24, 24))
        if heatmap.max() > 0:
            heatmap = heatmap/heatmap.max()
        else:
            heatmap = np.zeros((24, 24))+1
        heatmap = torch.tensor(heatmap).float()
        example_dict = {
            "id": example["id"],
            "encoder_images": image,
            "heatmap": heatmap,
            "direction": example["direction"],
            "labels": example["label"],
            'image_filepaths': example["image_file_path"][0],
        }
        if self.train and not self.self_critical:
            example_dict = {**example_dict, **self.tokenize(example["label"])}
        return example_dict
