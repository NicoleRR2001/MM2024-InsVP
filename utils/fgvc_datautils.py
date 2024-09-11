import math
import os

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL
from PIL import Image

# 声明全局变量 tmp
tmp = {}


class BaseJsonDataset(Dataset):
    def __init__(self, dataset, image_path, json_path, transform=None, mode='train'):
        self.transform = transform
        self.image_path = image_path
        self.split_json = json_path
        self.image_list = []
        self.label_list = []
        # 使用全局变量tmp
        global tmp
        with open(self.split_json) as fp:
            samples = json.load(fp)
            # print(samples)
            # print(samples[0])
            # input()
            for image, label in samples.items():
                # print(s)
                self.image_list.append(image)
                # 将label转换为int
                label = int(label)

                if dataset in ["cub", 'stanford_dogs', 'stanford_cars', 'flower102']:
                    label -= 1
                elif dataset == 'nabirds':
                    if label not in tmp:
                        tmp[label] = len(tmp)
                        # print(tmp)
                    label = tmp[label]
                self.label_list.append(label)
        # 将image_list和label_list打乱顺序
        if mode == 'train':
            tmp = list(zip(self.image_list, self.label_list))
            random.shuffle(tmp)
            self.image_list, self.label_list = zip(*tmp)
        # print(f"Dataset {dataset} has {len(self.image_list)} images.")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label).long()


root_path = "/data/dataset/liuzichen/fine-grained/"

path_dict = {
    # dataset_name: ["image_dir", "json_split_file"]
    "cub": ["CUB_200_2011/images/", "./utils/data_json/cub/"],
    "flower102": ["102flowers/", "./utils/data_json/oxfordflower/"],
    "nabirds": ["nabirds/images/", "./utils/data_json/nabirds/"],
    "stanford_dogs": ["stanford_dogs/Images/", "./utils/data_json/stanforddogs/"],
    "stanford_cars": ["stanford_cars/", "./utils/data_json/stanfordcars/"]
}

def build_dataset(dataset, mode='train', transform=None):
    path_suffix, json_path = path_dict[dataset.lower()]
    if mode == 'train':
        json_path = os.path.join(json_path, "train.json")
    elif mode == 'test':
        json_path = os.path.join(json_path, "test.json")
    image_path = os.path.join(root_path, path_suffix)
    return BaseJsonDataset(dataset, image_path, json_path, transform, mode=mode)
