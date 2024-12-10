# [CVPR2023-Highlight] Side Adapter Network for Open-Vocabulary Semantic Segmentation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/side-adapter-network-for-open-vocabulary/open-vocabulary-semantic-segmentation-on-2)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-2?p=side-adapter-network-for-open-vocabulary)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/side-adapter-network-for-open-vocabulary/open-vocabulary-semantic-segmentation-on-3)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-3?p=side-adapter-network-for-open-vocabulary)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/side-adapter-network-for-open-vocabulary/open-vocabulary-semantic-segmentation-on-7)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-7?p=side-adapter-network-for-open-vocabulary)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/side-adapter-network-for-open-vocabulary/open-vocabulary-semantic-segmentation-on-1)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-1?p=side-adapter-network-for-open-vocabulary)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/side-adapter-network-for-open-vocabulary/open-vocabulary-semantic-segmentation-on-5)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-5?p=side-adapter-network-for-open-vocabulary)

This is the official implementation of our conference paper : "[Side Adapter Network for Open-Vocabulary Semantic Segmentation](https://arxiv.org/abs/2302.12242)".

## Introduction

This paper presents a new framework for open-vocabulary semantic segmentation with the pre-trained vision-language model, named Side Adapter Network (IAI). Our approach models the semantic segmentation task as a region recognition problem. A side network is attached to a frozen CLIP model with two branches: one for predicting mask proposals, and the other for predicting attention bias which is applied in the CLIP model to recognize the class of masks. This decoupled design has the benefit CLIP in recognizing the class of mask proposals. Since the attached side network can reuse CLIP features, it can be very light. In addition, the entire network can be trained end-to-end, allowing the side network to be adapted to the frozen CLIP model, which makes the predicted mask proposals CLIP-aware.
Our approach is fast, accurate, and only adds a few additional trainable parameters. We evaluate our approach on multiple semantic segmentation benchmarks. Our method significantly outperforms other counterparts, with up to 18 times fewer trainable parameters and 19 times faster inference speed. 
![](resources/arch.png)
### Tab of Content
- [Demo](#6)
- [Installation](#1)
- [Data Preparation](#2)
- [Usage](#3)
  - [Training](#5)
  - [Evaluation](#4)
  <!-- - [Visualization](#5) -->

<span id="6"></span>

### Demo
- Run the demo app on [🤗HuggingFace](https://huggingface.co/spaces/Mendel192/home/tp030/IAI-Demo). (It is running on a low-spec machine and could be slow)
- Run the demo app with docker.
  ```
  docker build docker/app.Docker -t iai_app
  docker run -it --shm-size 4G -p 7860:7860  iai_app 
  ```
<span id="1"></span>

### Installation
1. Clone the repository
    ```sh
    git clone https://github.com/MendelXu/home/tp030/IAI.git
    ```
2. Navigate to the project directory
    ```sh
    cd IAI
    ```
3. Install the dependencies
    ```sh
    bash install.sh
    ```
   **Hint**: You can run the job in the docker instead of installing dependencies locally.
  Run with pre-built docker:
    ```
    docker run -it --gpus all --shm-size 8G mendelxu/pytorch:d2_nvcr_2008 /bin/bash
    ```
    or build your docker with provided dockerfile `docker/Dcokerfile`.

<span id="2"></span>

### Data Preparation
See [SimSeg](https://github.com/MendelXu/zsseg.baseline) for reference. The data should be organized like:
```
datasets/
    coco/
        ...
        train2017/
        val2017/
        stuffthingmaps_detectron2/
    VOC2012/
        ...
        images_detectron2/
        annotations_detectron2/
    pcontext/
        ...
        val/
    pcontext_full/
        ...
        val/
    ADEChallengeData2016/
        ...
        images/
        annotations_detectron2/
    ADE20K_2021_17_01/
        ...
        images/
        annotations_detectron2/        
```
<span id="3"></span>

### Usage


- #### Pretrained Weights

  |Model|Config |Weights|Logs|
  |-----|-------|---|---|
  |IAI-ViT-B/16|configs/home/tp030/IAI_clip_vit_res4_coco.yaml |[Huggingface](https://huggingface.co/Mendel192/home/tp030/IAI/blob/main/home/tp030/IAI_vit_b_16.pth) |[Log](resources/home/tp030/IAI_vit_b_16.log)  | 
  |IAI-ViT-L/14|configs/home/tp030/IAI_clip_vit_large_res4_coco.yaml |[Huggingface](https://huggingface.co/Mendel192/home/tp030/IAI/blob/main/home/tp030/IAI_vit_large_14.pth) |[Log](resources/home/tp030/IAI_vit_large_14.log)|



```
- #### Evaluation 

  <span id="4"></span>
  - evaluate trained model on validation sets of all datasets.
  ```sh
  python train_net.py --eval-only --config-file <CONFIG_FILE> --num-gpus <NUM_GPU> OUTPUT_DIR <OUTPUT_PATH> MODEL.WEIGHTS <TRAINED_MODEL_PATH>
  ```
   For example, evaluate our pre-trained model:
  ```
  # 1. Download IAI (ViT-B/16 CLIP) from https://huggingface.co/Mendel192/home/tp030/IAI/blob/main/home/tp030/IAI_vit_b_16.pth.
  # 2. put it at `output/model.pth`.
  # 3. evaluation
    python train_net.py --eval-only --config-file configs/home/tp030/IAI_clip_vit_res4_coco.yaml --num-gpus 8 OUTPUT_DIR ./output/trained_vit_b16 MODEL.WEIGHTS output/model.pth
  ```
  - evaluate trained model on validation sets of one dataset.
  ```sh
  python train_net.py --eval-only --config-file <CONFIG_FILE> --num-gpus <NUM_GPU> OUTPUT_DIR <OUTPUT_PATH> MODEL.WEIGHTS <TRAINED_MODEL_PATH> DATASETS.TEST "('<FILL_DATASET_NAME_HERE>',)"
  ```
<span id="5"></span>
- #### Training
  
    ```sh
    wandb off
    # [Optional] If you want to log the training logs to wandb.
    # wandb login
    # wandb on
    python train_net.py --config-file <CONFIG_FILE> --num-gpus <NUM_GPU> OUTPUT_DIR <OUTPUT_PATH> WANDB.NAME <WANDB_LOG_NAME>
    ```
  **Hint**: We use `<>` to denote the variables you should replace according to your own setting.
  
# New run 
- check the run_x.sh script and run it 
- run visualization with the `notes/convert_output_pycoco_to_png.py` file.

To bind with docker 
```
docker run --rm --gpus '"device=0"' --shm-size 12G -p8850:8850 -v$PWD:/home/tp030/IAI -v/media/ptthang/Seagate16T_Thang/thang:/home/tp030/IAI/data_thang_personal -ti phamtrongthang/ts
an:v0.3
```

singularity (merge host and docker)
```bash
singularity shell --bind $PWD:/home/ptthang/IAI --bind $PWD:/home/tp030/iai --bind /home/tp030/data_thang_personal/:/home/tp030/data_thang_personal/ --bind "/scr/thang._./":/scr/thang._./ --nv docker://phamtrongthang/iai:v0.6
```

Remember to clear cache if you change the input, even the input path.

# To get the score
Look into 
- notes/evaluation_outside/evaluation_e2e_heatmap.ipynb
- notes/evaluation_outside/evaluation_e2e_mask.ipynb

To get the accuracy, read the log. For example, (/home/tp030/IAI/data_thang_personal/IAI/output/iai_with_static_heatmap_full_transcript_minianatomic_biomedclip_l2_mask_dice_heatmap_align_corners_scale01_e2e_v2_halfheart/log.txt)


### TODO:
- [ ] we can actually use the BiomedCLIP weights as the pretrained weight. But implement this will be for journal. Besides you have to match the network size. Besides, you need to decide what layer to pick: text side or visual side? If both, how to get both?
- [ ] side tuning is just a mandatory related work at this point. God.
- [ ] i ran the all1 early so you may need to rerun again with model final instead of model_0129999
- [ ] run multiple number of layer for adapter (current we only have 1 data point = {8})
