from torchvision import transforms
import torch
import torchvision
import numpy as np

from utils import fgvc_datautils
from utils import vtab_datautils

ViT_input_transforms_default = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    # transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ViT_input_transforms_default_train = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ViT_input_transforms_default_test = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ViT_input_transforms_SOTA_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ViT_input_transforms_SOTA_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset(args, mode='train'):
    if args.transform == 'default':
        if mode == 'train':
            ViT_input_transforms = ViT_input_transforms_default_train
        else:
            ViT_input_transforms = ViT_input_transforms_default_test
    elif args.transform == 'SOTA':
        if mode == 'train':
            ViT_input_transforms = ViT_input_transforms_SOTA_train
        else:
            ViT_input_transforms = ViT_input_transforms_SOTA_test
            
    if args.dataset in ['cub', 'nabirds', 'stanford_dogs', 'stanford_cars', 'flower102']:
        dataset = fgvc_datautils.build_dataset(args.dataset, mode=mode, transform=ViT_input_transforms)
    elif args.dataset in ['vtab-caltech101', 'vtab-cifar100', 'vtab-dtd', 'vtab-food101', 
                          'vtab-flower', 'vtab-sun397', 'vtab-clevr(task="closest_object_distance")',
                            'vtab-clevr(task="count_all")', 
                            'vtab-smallnorb(predicted_attribute="label_azimuth")',
                            'vtab-smallnorb(predicted_attribute="label_elevation")',
                            'vtab-dsprites(predicted_attribute="label_x_position",num_classes=16)',
                            'vtab-dsprites(predicted_attribute="label_orientation",num_classes=16)',
                            'vtab-kitti', 'vtab-dmlab',
                         'vtab-FGVCAircraft', 'vtab-eurosat', 'vtab-pets', 'vtab-svhn', 'vtab-GTSRB']:
        dataset = vtab_datautils.build_dataset(args.dataset, mode=mode, transform=ViT_input_transforms)
        
    if args.dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100('/data/dataset/liuzichen/torchvision/Cifar100',
                                                 train=(mode == 'train'), 
                                                 transform=ViT_input_transforms, 
                                                 download=True)
    elif args.dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10('/data/dataset/liuzichen/torchvision/Cifar10',
                                                    train=(mode == 'train'), 
                                                    transform=ViT_input_transforms, 
                                                    download=True)
    # elif args.dataset == 'flower102':
    #     dataset = torchvision.datasets.Flowers102('/data/dataset/liuzichen/torchvision/Flower102',
    #                                                 split=mode, 
    #                                                 transform=ViT_input_transforms, 
    #                                                 download=True)
    elif args.dataset == 'food101':
        dataset = torchvision.datasets.Food101('/data/dataset/liuzichen/torchvision/Food101',
                                                    split=mode, 
                                                    transform=ViT_input_transforms, 
                                                    download=True)
    elif args.dataset == 'FGVCAircraft':
        dataset = torchvision.datasets.FGVCAircraft('/data/dataset/liuzichen/torchvision/FGVCAircraft',
                                                    split=mode, 
                                                    transform=ViT_input_transforms, 
                                                    download=True)
    elif args.dataset == 'EuroSAT':
        dataset = torchvision.datasets.EuroSAT('/data/dataset/liuzichen/torchvision/EuroSAT',
                                                    train=(mode == 'train'), 
                                                    transform=ViT_input_transforms, 
                                                    download=True)
    elif args.dataset == 'OxfordIIITPet':
        dataset = torchvision.datasets.OxfordIIITPet('/data/dataset/liuzichen/torchvision/OxfordIIITPet',
                                                    train=(mode == 'train'), 
                                                    transform=ViT_input_transforms, 
                                                    download=True)
    elif args.dataset == 'DTD':
        dataset = torchvision.datasets.DTD('/data/dataset/liuzichen/torchvision/DTD',
                                                    split=mode,
                                                    transform=ViT_input_transforms, 
                                                    download=True)
    elif args.dataset == 'SVHN':
        dataset = torchvision.datasets.SVHN('/data/dataset/liuzichen/torchvision/SVHN',
                                                    split=mode, 
                                                    transform=ViT_input_transforms, 
                                                    download=True)
    elif args.dataset == 'GTSRB':
        dataset = torchvision.datasets.GTSRB('/data/dataset/liuzichen/torchvision/GTSRB',
                                                    split=mode, 
                                                    transform=ViT_input_transforms, 
                                                    download=True)
    # elif args.dataset == 'stanford_cars':
    #     dataset = torchvision.datasets.StanfordCars('/data/dataset/liuzichen/fine-grained/StanfordCars',
    #                                                 split=mode, 
    #                                                 transform=ViT_input_transforms, 
    #                                                 download=False)
    #                                                 # download=True)
    return dataset


# for schedule

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def get_ECOC_code(N_class, N_bit):
    # 对于 N_class 个类别，使用 N_bit 位的 ECOC 编码
    # 计算有几位冗余
    N_redundant = N_bit - int(np.ceil(np.log2(N_class)))
    # 生成 N_bit 位的 ECOC 编码
    ECOC_code = np.random.randint(0, 2, size=(N_class, N_bit))
    # 生成冗余位
    for i in range(N_redundant):
        ECOC_code[:, -i-1] = np.logical_xor.reduce(ECOC_code[:, -i-1:-N_redundant+i], axis=1)
    # 将 ECOC 编码转换为 -1 和 1
    ECOC_code = np.where(ECOC_code == 0, -1, 1)
    # print(f"ECOC_code: {ECOC_code}")
    return ECOC_code
    