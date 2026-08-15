import json
import os

import open_clip
from detectron2.data import DatasetCatalog, MetadataCatalog
from torchvision import transforms


def IAI_gaze_function(df, root):
    # change this function into for looping all image type, if there is left then add left item, if there is right then add right item, if there is cardiac then add cardiac item, and finally add the static heatmap item
    root_image_path = os.path.join(root, "images")
    root_mask_path = os.path.join(root, "masks_from_heatmaps")
    root_heatmap_path = os.path.join(root, "heatmaps")

    d_dicts = []
    for k, v in df.items():
        for i, type in enumerate(v["type"]):
            type_ = type.replace(" ", "_")
            res = {}
            # only jpg use dicom_id, the rest use key
            dicom_id = v["dicom_id"]
            image_jpg = f"{dicom_id}.jpg"
            res["image_jpg"] = os.path.join(root_image_path, image_jpg)
            res["gt_heatmap_path"] = os.path.join(root_heatmap_path, f"{k}_{type_}.png")
            res["gt_mask_path"] = os.path.join(root_mask_path, f"{k}_{type_}.png")
            res["dicom_id"] = dicom_id
            res["heatmap_type"] = type_
            res["transcript"] = type
            res["label"] = 0
            d_dicts.append(res)

    return d_dicts


def register_all_iai(root):
    dset = "debug"
    dset = "full"
    annotation_path = os.path.join(root, f"{dset}.json")
    with open(annotation_path) as handle:
        train_df = json.load(handle)
    test_df = train_df

    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    input_size = preprocess_train.transforms[0].size[0]
    preprocess_heatmap = transforms.Compose(
        [transforms.Resize([input_size, input_size]), transforms.ToTensor()]
    )
    for name, df, preprocess in [
        ("train", train_df, preprocess_train),
        ("test", test_df, preprocess_val),
    ]:
        data_name = f"IAI_gaze_{name}_e2e_v2"
        DatasetCatalog.register(
            data_name,
            lambda x=df: IAI_gaze_function(x, root),
        )
        MetadataCatalog.get(data_name).set(
            label_root=annotation_path,
            evaluator_type="IAI_gaze_e2e_v2",
            ignore_label=255,
            thing_classes=["NotInterest", "Interest"],
            preprocess=preprocess,
            preprocess_heatmap=preprocess_heatmap,
        )


_root = os.getenv("DETECTRON2_DATASETS", "data/fg_cxr")
register_all_iai(_root)
