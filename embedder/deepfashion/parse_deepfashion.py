import os
import json
import tqdm
import typing as tp

from collections import defaultdict
from dataclasses import dataclass
from PIL import Image


@dataclass
class Crop:
    image_id: int
    full_image_file: str
    category_id: int
    category_name: str
    item_style: int
    bbox: tp.List[int]
    source: str
    crop_file: tp.Optional[str] = None


def valid_bbox(bbox) -> bool:
    if bbox[2] <= bbox[0]:
        return False
    if bbox[3] <= bbox[1]:
        return False
    for coord in bbox:
        if coord < 0:
            return False
    return True


def save_crop_file(dir, split, crop: Crop):
    Image.open(crop.full_image_file).crop(crop.bbox).save(crop.crop_file)


def read_items(dir, split, min_item_id = 0, generate_crops=False, main_only=True):
    item2crops = defaultdict(list)
    for filename in tqdm.tqdm(os.listdir(f"{dir}/{split}/annos")):
        with open(f"{dir}/{split}/annos/{filename}", "r") as f:
            annotation = json.load(f)
        image_id = filename.split(".")[0]
        full_image_file = f"{dir}/{split}/image/{image_id}.jpg"
        item_id = int(annotation["pair_id"]) + min_item_id
        source = annotation["source"]
        bbox_id = 1
        while f"item{bbox_id}" in annotation:
            key = f"item{bbox_id}"
            item_style = annotation[key]['style']
            if main_only:
                if item_style == 0:
                    bbox_id += 1
                    continue   
            category_id = annotation[key]['category_id']
            category_name = annotation[key]['category_name']
            bbox = annotation[key]['bounding_box']
            if main_only:
                crop_file = f"{dir}/{split}/crops/{image_id}.jpg"
            else:
                crop_file = f"{dir}/{split}/crops/{image_id}_{bbox_id}.jpg"
            crop = Crop(image_id, full_image_file, category_id, category_name,
                        item_style, bbox, source, crop_file)
            if valid_bbox(bbox):
                item2crops[item_id].append(crop)
            if main_only:
                break
            bbox_id += 1
    if generate_crops:
        os.makedirs(f"{dir}/{split}/crops", exist_ok=True)
        for crops in tqdm.tqdm(item2crops.values()):
            for crop in crops:
                save_crop_file(dir, split, crop)
    return item2crops


def read_splits(dir: str, splits: tp.List[str], generate_crops=False, main_only=True):
    res  = {-1: None}
    for split in splits:
        print(f"Reading {split} split")
        min_item_id = max(res.keys()) + 1
        res.update(read_items(dir, split, min_item_id, generate_crops, main_only))
    del res[-1]
    return res
