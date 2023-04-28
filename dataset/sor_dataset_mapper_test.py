import copy
import cv2
import numpy as np
from PIL import Image
import albumentations as A
import torch

def read_image(file_name, format="RGB"):
    return np.array(Image.open(file_name).convert(format)).astype(np.uint8)

def parse_anno(anno, H, W):
    mask = np.zeros((H,W), dtype=float)
    cv2.fillPoly(mask, [np.array(xy).reshape(-1,2) for xy in anno["segmentation"]], 1.0)
    return mask

def merge_masks(masks, H, W):
    mask = np.zeros((H,W), dtype=float)
    for m in masks:
        mask += m
    return np.where(mask>0.5, 1.0, 0.0)

def sor_dataset_mapper_test(dataset_dict, cfg):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = read_image(dataset_dict["file_name"], format="RGB")
    H, W = dataset_dict["height"], dataset_dict["width"]
    ranks = [anno["category_id"] for anno in dataset_dict["annotations"]]
    order_of_ranks = np.argsort(ranks)[::-1]
    masks = [parse_anno(dataset_dict["annotations"][i], H, W) for i in order_of_ranks if ranks[i]>0]
    ior_mask = np.zeros((H,W))

    ## data aug: only resize is adopted for inference stage
    transform = A.Compose([
            A.Resize(cfg.INPUT.FT_SIZE_TEST, cfg.INPUT.FT_SIZE_TEST)
        ],
        additional_targets = {"ior_mask": "mask"}
    )
    aug = transform(image = image, ior_mask=ior_mask)
    image, ior_mask = aug["image"], aug["ior_mask"]

    ## toTensor
    image = torch.from_numpy(image).permute(2,0,1).float()
    ior_mask = torch.from_numpy(ior_mask).float()
    masks = [ torch.from_numpy(mask).float() for mask in masks]

    return {
        "image_name": dataset_dict["image_name"],
        "image": image,
        "height": dataset_dict["height"],
        "width": dataset_dict["width"],
        "ior_mask": ior_mask,

        "masks": masks, ## GT for the purpose of evaluation,
        "ranks": np.array(ranks)[order_of_ranks].tolist() ## for verification
    }
