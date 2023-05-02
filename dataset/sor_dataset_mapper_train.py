import copy
import cv2
import numpy as np
import torch
from PIL import Image
import albumentations as A


def read_image(file_name, format="RGB"):
    return np.array(Image.open(file_name).convert(format)).astype(np.uint8)


def parse_anno(anno, H, W):
    mask = np.zeros((H, W), dtype=float)
    cv2.fillPoly(mask, [np.array(xy).reshape(-1, 2) for xy in anno["segmentation"]], 1.0)
    return mask


def merge_masks(masks, H, W):
    mask = np.zeros((H, W), dtype=float)
    for m in masks:
        mask += m
    return np.where(mask > 0.5, 1.0, 0.0)


def sampleRankExp(ranks):
    num_level = max(ranks)
    target = int(np.log2(np.random.randint(0, int(2 ** num_level)) + 1)) + 1  ## random sample
    target = 0 if target > num_level else target  ## if target>num_level means we mask all sal objs
    return target

def sampleRankUni(ranks):
    return max(ranks)
    # ables = [r for r in ranks if r > 0] + [0]
    # return ables[np.random.randint(0, len(ables))] if len(ables) > 0 else 0

def sor_dataset_mapper_train(dataset_dict, cfg):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = read_image(dataset_dict["file_name"], format="RGB")
    H, W = dataset_dict["height"], dataset_dict["width"]
    cates = []
    masks = []
    for anno in dataset_dict["annotations"]:
        cate = anno["category_id"]
        if cate > 0:
            cates.append(cate)
            masks.append(parse_anno(anno, H, W))

    ## data aug
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Resize(cfg.INPUT.FT_SIZE_TRAIN, cfg.INPUT.FT_SIZE_TRAIN)
        # A.LongestMaxSize(max_size=cfg.INPUT.FT_SIZE_TRAIN),
        # A.PadIfNeeded(min_height=cfg.INPUT.FT_SIZE_TRAIN, min_width=cfg.INPUT.FT_SIZE_TRAIN)
    ],
        additional_targets=dict(("mask_{}".format(i), "mask") for i in range(len(masks)))
    )
    aug = transform(image=image, **dict(("mask_{}".format(i), masks[i]) for i in range(len(masks))))
    image = aug["image"]
    masks = [aug["mask_{}".format(i)] for i in range(len(masks))]

    ## toTensor
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    masks = [torch.from_numpy(m).float() for m in masks]

    ## selection
    target = torch.zeros(image.shape[-2::])
    target_rank = sampleRankUni(cates)
    ior_masks = []
    ior_ranks = []
    for r, m in zip(cates, masks):
        if r > target_rank:
            ior_masks.append(m)
            ior_ranks.append(r)
        elif r == target_rank and r > 0:
            target = m.clone()

    return {
        "image_name": dataset_dict["image_name"],
        "image": image,
        "height": H,
        "width": W,
        "target": target,
        "target_rank": target_rank,
        "ior_masks": ior_masks,
        "ior_ranks": ior_ranks
    }