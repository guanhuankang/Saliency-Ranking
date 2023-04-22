import copy
import cv2
import numpy as np
from PIL import Image
import albumentations as A

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

def sampleRank(ranks):
    num_level = max(ranks)
    target = int(np.log2(np.random.randint(0, int(2**num_level)) + 1)) + 1 ## random sample
    target = 0 if target>num_level else target ## if target>num_level means we mask all sal objs
    return target

def sor_dataset_mapper(dataset_dict, cfg):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = read_image(dataset_dict["file_name"], format="RGB")
    ranks = [anno["category_id"] for anno in dataset_dict["annotations"]]
    target = sampleRank(ranks)
    H, W = dataset_dict["height"], dataset_dict["width"]

    ior_mask = merge_masks([ 
        parse_anno(dataset_dict["annotations"][i], H, W) 
        for i in range(len(ranks)) if ranks[i]>target
    ], H, W)

    mask = merge_masks([ 
        parse_anno(dataset_dict["annotations"][i], H, W) 
        for i in range(len(ranks)) if (ranks[i]==target and target>0)
    ], H, W)
    
    ## data aug
    transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Resize(cfg.INPUT.FT_SIZE_TRAIN, cfg.INPUT.FT_SIZE_TRAIN)
        ],
        additional_targets={
            "ior_mask": "mask"
        }
    )
    aug = transform(image = image, ior_mask=ior_mask, mask=mask)
    image, ior_mask, mask = aug["image"], aug["ior_mask"], aug["mask"]

    return {
        "image_name": dataset_dict["image_name"],
        "image": image,
        "height": dataset_dict["height"],
        "width": dataset_dict["width"],
        "image_id": dataset_dict["image_id"],
        "ior_mask": ior_mask,
        "mask": mask,
        "score": 1.0 if target>0 else 0.0,
        "rank": target,
        "ranks": ranks
    }
