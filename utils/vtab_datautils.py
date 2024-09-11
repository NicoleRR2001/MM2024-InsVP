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
    def __init__(self, dataset, image_path, json_path, transform=None, mode='train', task = None):
        self.dataset = dataset
        self.transform = transform
        self.image_path = image_path
        self.split_json = json_path
        self.image_list = []
        self.label_list = []
        self.dataset_wo_path = ['svhn', 'caltech101', 'cifar100', 
                                'clevr', 'smallnorb', 'dsprites', 'kitti', 'dmlab']
        # 使用全局变量tmp
        global tmp
        with open(self.split_json) as fp:
            samples = json.load(fp)
            # print(samples)
            # print(samples[0])
            # input()
            for sample in samples:
                if task is None:
                    image, label = sample
                elif task in ['count_all', 'label_azimuth', 'label_x_position']:
                    image, label, _ = sample
                elif task in ['closest_object_distance', 'label_elevation', 'label_orientation']:
                    image, _, label = sample
                
                    
                # print(s)
                if dataset in self.dataset_wo_path:
                    if mode == 'train':
                        image = os.path.join("train800val200", image)
                    elif mode == 'test':
                        image = os.path.join("test", image)
                if dataset == 'eurosat':
                    image = os.path.join(image.split("_")[0], image)
                self.image_list.append(image)
                # print(image)
                # input()
                # 将label转换为int
                label = int(label)
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
        # print(self.image_list[idx])
        # print(self.image_path)
        if self.dataset in ['sun397']:
            image_path = self.image_path + self.image_list[idx]
        else:
            image_path = os.path.join(self.image_path, self.image_list[idx])
        # print(image_path)
        # input()
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label).long()


root_path = "/data/dataset/liuzichen/fine-grained/"

dataset_wo_path = ['svhn', 'caltech101', 'cifar100', 
                   'clevr', 'smallnorb', 'dsprites', 'kitti', 'dmlab']
root_path_2 = "/data/dataset/liuzichen/VTAB/"

path_dict = {
    # dataset_name: ["image_dir", "json_split_file"]
    # "Caltech
    "caltech101": ["caltech101/", "./utils/data_json/vtab/caltech101"],
    "cifar100": ["cifar100/", "./utils/data_json/vtab/cifar100"],
    "dtd": ["DTD/images/", "./utils/data_json/vtab/dtd"],
    "flower": ["102flowers/jpg", "./utils/data_json/vtab/flower"],
    "pets": ["OxfordPets/images/", "./utils/data_json/vtab/pets"],
    "sun397": ["SUN397", "./utils/data_json/vtab/sun397"],
    "svhn": ["svhn/", "./utils/data_json/vtab/svhn"],
    
    # 
    'eurosat': ['EuroSAT', './utils/data_json/vtab/eurosat'],
    
    'clevr(task="closest_object_distance")': ['clevr/', "./utils/data_json/vtab/clevr"],
    'clevr(task="count_all")': ['clevr/', "./utils/data_json/vtab/clevr"],
    'smallnorb(predicted_attribute="label_azimuth")': ['smallnorb/', "./utils/data_json/vtab/smallnorb"],
    'smallnorb(predicted_attribute="label_elevation")': ['smallnorb/', "./utils/data_json/vtab/smallnorb"],
    'dsprites(predicted_attribute="label_x_position",num_classes=16)': ['dsprites/', "./utils/data_json/vtab/dsprites"],
    'dsprites(predicted_attribute="label_orientation",num_classes=16)': ['dsprites/', "./utils/data_json/vtab/dsprites"],
    'kitti': ['kitti/', "./utils/data_json/vtab/kitti"],
    'dmlab': ['dmlab/', "./utils/data_json/vtab/dmlab"],
    
}

def build_dataset(dataset, mode='train', transform=None):
    task = None
    dataset = dataset.split("-")[1]
    path_suffix, json_path = path_dict[dataset.lower()]
    if dataset == 'clevr(task="closest_object_distance")':
        dataset = 'clevr'
        task = 'closest_object_distance'
    elif dataset == 'clevr(task="count_all")':
        dataset = 'clevr'
        task = 'count_all'
    elif dataset == 'smallnorb(predicted_attribute="label_azimuth")':
        dataset, task = 'smallnorb', 'label_azimuth'
    elif dataset == 'smallnorb(predicted_attribute="label_elevation")':
        dataset, task = 'smallnorb', 'label_elevation'
    elif dataset == 'dsprites(predicted_attribute="label_x_position",num_classes=16)':
        dataset, task = 'dsprites', 'label_x_position'
    elif dataset == 'dsprites(predicted_attribute="label_orientation",num_classes=16)':
        dataset, task = 'dsprites', 'label_orientation'
    if mode == 'train':
        json_path = os.path.join(json_path, dataset+"_train800val200.json")
    elif mode == 'test':
        json_path = os.path.join(json_path, dataset+"_test.json")
    if dataset in dataset_wo_path:
        image_path = os.path.join(root_path_2, path_suffix)
    else:
        image_path = os.path.join(root_path, path_suffix)
    # print(image_path)
    # input()
    return BaseJsonDataset(dataset, image_path, json_path, transform, mode=mode, task=task)


# class Aircraft(Dataset):
#     """ FGVC Aircraft dataset """
#     def __init__(self, root, mode='train', n_shot=None, transform=None):
#         self.transform = transform
#         self.path = root
#         self.mode = mode

#         self.cname = []
#         with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
#             self.cname = [l.replace("\n", "") for l in fp.readlines()]

#         self.image_list = []
#         self.label_list = []
#         with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
#             lines = [s.replace("\n", "") for s in fp.readlines()]
#             for l in lines:
#                 ls = l.split(" ")
#                 img = ls[0]
#                 label = " ".join(ls[1:])
#                 self.image_list.append("{}.jpg".format(img))
#                 self.label_list.append(self.cname.index(label))

#         if n_shot is not None:
#             few_shot_samples = []
#             c_range = max(self.label_list) + 1
#             for c in range(c_range):
#                 c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
#                 random.seed(0)
#                 few_shot_samples.extend(random.sample(c_idx, n_shot))
#             self.image_list = [self.image_list[i] for i in few_shot_samples]
#             self.label_list = [self.label_list[i] for i in few_shot_samples]

#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.path, 'images', self.image_list[idx])
#         image = Image.open(image_path).convert('RGB')
#         label = self.label_list[idx]
#         if self.transform:
#             image = self.transform(image)
        
#         return image, torch.tensor(label).long()

