from torchvision.models import vit_b_16, ViT_B_16_Weights, resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
import torch.nn as nn
import torch
from timm.models import create_model
import timm



class SimamModule(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-3):
        super(SimamModule, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


Dataset_N_classes = {'cifar100': 100, 
                     'cifar10': 10, 
                     'flower102': 102, 
                     'food101': 101, 
                     'FGVCAircraft': 100,
                     'EuroSAT': 10,
                     'OxfordIIITPet': 37,
                     'DTD': 47,
                     'dtd': 47,
                     'SVHN': 10,
                     'svhn': 10,
                     'GTSRB': 43,
                     'gtsrb': 43,
                     'stanford_cars': 196,
                     'cub': 200,
                     'nabirds': 555,
                     'stanford_dogs': 120,
                     'vtab-cifar(num_classes=100)':100,
                     'vtab-dtd': 47,
                     'vtab-flower': 102,
                     "vtab-pets": 37,
                     'vtab-svhn': 10,
                     'vtab-sun397': 397,
                     'vtab-caltech101': 102,
                     'vtab-cifar100':100,
                     'vtab-eurosat': 10,
                     'vtab-clevr(task="closest_object_distance")': 6,
                     'vtab-clevr(task="count_all")': 8,
                     'vtab-smallnorb(predicted_attribute="label_azimuth")': 18,
                     'vtab-smallnorb(predicted_attribute="label_elevation")': 9,
                     'vtab-dsprites(predicted_attribute="label_x_position",num_classes=16)': 16,
                     'vtab-dsprites(predicted_attribute="label_orientation",num_classes=16)': 16,
                     'vtab-kitti': 4,
                     'vtab-dmlab': 6,
}

def get_backbone(args):
    
    if args.pretrained == 'imagenet1k':
        if args.arch == 'ViT/B-16':
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            backbone = vit_b_16(weights=weights)
        elif args.arch == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1
            backbone = resnet50(weights=weights)
        elif args.arch == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1
            backbone = resnet18(weights=weights)
        
    elif args.pretrained == 'imagenet22k':
        if args.arch == 'ViT/B-16':
            backbone = create_model(
                # 'vit_base_patch16_224_in21k',
                "vit_base_patch16_224.augreg_in21k",
                pretrained=False,
                num_classes=21843,
                drop_block_rate=None,
            )
            weight_path = '../~/models/imagenet-22k/vit_base_p16_224_in22k.pth'
            backbone.load_state_dict(torch.load(weight_path, weights_only=True), False)
        


    N_classes = Dataset_N_classes[args.dataset]

    if args.pretrained == 'imagenet22k':
        if args.arch in ['ViT/B-16', 'swin']:
            if args.cls == "default":
                backbone.reset_classifier(num_classes=N_classes)
        elif args.arch == 'resnet50':
            backbone.fc = nn.Linear(2048, N_classes)
        elif args.arch == 'resnet18':
            backbone.fc = nn.Linear(512, N_classes)
    return backbone


Weight_transform = {'cls_token': 'class_token',
                    'norm.bias': 'encoder.ln.bias',
                    'norm.weight': 'encoder.ln.weight',
                    'blocks.0.norm1.bias': 'encoder.layers.encoder_layer_0.ln_1.bias',
                    'blocks.0.norm1.weight': 'encoder.layers.encoder_layer_0.ln_1.weight',
                    'blocks.0.norm2.bias': 'encoder.layers.encoder_layer_0.ln_2.bias',
                    'blocks.0.norm2.weight': 'encoder.layers.encoder_layer_0.ln_2.weight',
                    'blocks.0.mlp.fc1.bias': 'encoder.layers.encoder_layer_0.mlp.0.bias',
                    'blocks.0.mlp.fc1.weight': 'encoder.layers.encoder_layer_0.mlp.0.weight',
                    'blocks.0.mlp.fc2.bias': 'encoder.layers.encoder_layer_0.mlp.3.bias',
                    'blocks.0.mlp.fc2.weight': 'encoder.layers.encoder_layer_0.mlp.3.weight',
                    'blocks.0.attn.proj.bias': 'encoder.layers.encoder_layer_0.self_attention.out_proj.bias',
                    'blocks.0.attn.proj.weight': 'encoder.layers.encoder_layer_0.self_attention.out_proj.weight',
                    'blocks.0.attn.qkv.bias': 'encoder.layers.encoder_layer_0.self_attention.in_proj_bias',
                    'blocks.0.attn.qkv.weight': 'encoder.layers.encoder_layer_0.self_attention.in_proj_weight',
                    'blocks.1.norm1.bias': 'encoder.layers.encoder_layer_1.ln_1.bias',
                    'blocks.1.norm1.weight': 'encoder.layers.encoder_layer_1.ln_1.weight',
                    'blocks.1.norm2.bias': 'encoder.layers.encoder_layer_1.ln_2.bias',
                    'blocks.1.norm2.weight': 'encoder.layers.encoder_layer_1.ln_2.weight',
                    'blocks.1.mlp.fc1.bias': 'encoder.layers.encoder_layer_1.mlp.0.bias',
                    'blocks.1.mlp.fc1.weight': 'encoder.layers.encoder_layer_1.mlp.0.weight',
                    'blocks.1.mlp.fc2.bias': 'encoder.layers.encoder_layer_1.mlp.3.bias',
                    'blocks.1.mlp.fc2.weight': 'encoder.layers.encoder_layer_1.mlp.3.weight',
                    'blocks.1.attn.proj.bias': 'encoder.layers.encoder_layer_1.self_attention.out_proj.bias',
                    'blocks.1.attn.proj.weight': 'encoder.layers.encoder_layer_1.self_attention.out_proj.weight',
                    'blocks.1.attn.qkv.bias': 'encoder.layers.encoder_layer_1.self_attention.in_proj_bias',
                    'blocks.1.attn.qkv.weight': 'encoder.layers.encoder_layer_1.self_attention.in_proj_weight',
                    'blocks.2.norm1.bias': 'encoder.layers.encoder_layer_2.ln_1.bias',
                    'blocks.2.norm1.weight': 'encoder.layers.encoder_layer_2.ln_1.weight',
                    'blocks.2.norm2.bias': 'encoder.layers.encoder_layer_2.ln_2.bias',
                    'blocks.2.norm2.weight': 'encoder.layers.encoder_layer_2.ln_2.weight',
                    'blocks.2.mlp.fc1.bias': 'encoder.layers.encoder_layer_2.mlp.0.bias',
                    'blocks.2.mlp.fc1.weight': 'encoder.layers.encoder_layer_2.mlp.0.weight',
                    'blocks.2.mlp.fc2.bias': 'encoder.layers.encoder_layer_2.mlp.3.bias',
                    'blocks.2.mlp.fc2.weight': 'encoder.layers.encoder_layer_2.mlp.3.weight',
                    'blocks.2.attn.proj.bias': 'encoder.layers.encoder_layer_2.self_attention.out_proj.bias',
                    'blocks.2.attn.proj.weight': 'encoder.layers.encoder_layer_2.self_attention.out_proj.weight',
                    'blocks.2.attn.qkv.bias': 'encoder.layers.encoder_layer_2.self_attention.in_proj_bias',
                    'blocks.2.attn.qkv.weight': 'encoder.layers.encoder_layer_2.self_attention.in_proj_weight',
                    'blocks.3.norm1.bias': 'encoder.layers.encoder_layer_3.ln_1.bias',
                    'blocks.3.norm1.weight': 'encoder.layers.encoder_layer_3.ln_1.weight',
                    'blocks.3.norm2.bias': 'encoder.layers.encoder_layer_3.ln_2.bias',
                    'blocks.3.norm2.weight': 'encoder.layers.encoder_layer_3.ln_2.weight',
                    'blocks.3.mlp.fc1.bias': 'encoder.layers.encoder_layer_3.mlp.0.bias',
                    'blocks.3.mlp.fc1.weight': 'encoder.layers.encoder_layer_3.mlp.0.weight',
                    'blocks.3.mlp.fc2.bias': 'encoder.layers.encoder_layer_3.mlp.3.bias',
                    'blocks.3.mlp.fc2.weight': 'encoder.layers.encoder_layer_3.mlp.3.weight',
                    'blocks.3.attn.proj.bias': 'encoder.layers.encoder_layer_3.self_attention.out_proj.bias',
                    'blocks.3.attn.proj.weight': 'encoder.layers.encoder_layer_3.self_attention.out_proj.weight',
                    'blocks.3.attn.qkv.bias': 'encoder.layers.encoder_layer_3.self_attention.in_proj_bias',
                    'blocks.3.attn.qkv.weight': 'encoder.layers.encoder_layer_3.self_attention.in_proj_weight',
                    'blocks.4.norm1.bias': 'encoder.layers.encoder_layer_4.ln_1.bias',
                    'blocks.4.norm1.weight': 'encoder.layers.encoder_layer_4.ln_1.weight',
                    'blocks.4.norm2.bias': 'encoder.layers.encoder_layer_4.ln_2.bias',
                    'blocks.4.norm2.weight': 'encoder.layers.encoder_layer_4.ln_2.weight',
                    'blocks.4.mlp.fc1.bias': 'encoder.layers.encoder_layer_4.mlp.0.bias',
                    'blocks.4.mlp.fc1.weight': 'encoder.layers.encoder_layer_4.mlp.0.weight',
                    'blocks.4.mlp.fc2.bias': 'encoder.layers.encoder_layer_4.mlp.3.bias',
                    'blocks.4.mlp.fc2.weight': 'encoder.layers.encoder_layer_4.mlp.3.weight',
                    'blocks.4.attn.proj.bias': 'encoder.layers.encoder_layer_4.self_attention.out_proj.bias',
                    'blocks.4.attn.proj.weight': 'encoder.layers.encoder_layer_4.self_attention.out_proj.weight',
                    'blocks.4.attn.qkv.bias': 'encoder.layers.encoder_layer_4.self_attention.in_proj_bias',
                    'blocks.4.attn.qkv.weight': 'encoder.layers.encoder_layer_4.self_attention.in_proj_weight',
                    'blocks.5.norm1.bias': 'encoder.layers.encoder_layer_5.ln_1.bias',
                    'blocks.5.norm1.weight': 'encoder.layers.encoder_layer_5.ln_1.weight',
                    'blocks.5.norm2.bias': 'encoder.layers.encoder_layer_5.ln_2.bias',
                    'blocks.5.norm2.weight': 'encoder.layers.encoder_layer_5.ln_2.weight',
                    'blocks.5.mlp.fc1.bias': 'encoder.layers.encoder_layer_5.mlp.0.bias',
                    'blocks.5.mlp.fc1.weight': 'encoder.layers.encoder_layer_5.mlp.0.weight',
                    'blocks.5.mlp.fc2.bias': 'encoder.layers.encoder_layer_5.mlp.3.bias',
                    'blocks.5.mlp.fc2.weight': 'encoder.layers.encoder_layer_5.mlp.3.weight',
                    'blocks.5.attn.proj.bias': 'encoder.layers.encoder_layer_5.self_attention.out_proj.bias',
                    'blocks.5.attn.proj.weight': 'encoder.layers.encoder_layer_5.self_attention.out_proj.weight',
                    'blocks.5.attn.qkv.bias': 'encoder.layers.encoder_layer_5.self_attention.in_proj_bias',
                    'blocks.5.attn.qkv.weight': 'encoder.layers.encoder_layer_5.self_attention.in_proj_weight',
                    'blocks.6.norm1.bias': 'encoder.layers.encoder_layer_6.ln_1.bias',
                    'blocks.6.norm1.weight': 'encoder.layers.encoder_layer_6.ln_1.weight',
                    'blocks.6.norm2.bias': 'encoder.layers.encoder_layer_6.ln_2.bias',
                    'blocks.6.norm2.weight': 'encoder.layers.encoder_layer_6.ln_2.weight',
                    'blocks.6.mlp.fc1.bias': 'encoder.layers.encoder_layer_6.mlp.0.bias',
                    'blocks.6.mlp.fc1.weight': 'encoder.layers.encoder_layer_6.mlp.0.weight',
                    'blocks.6.mlp.fc2.bias': 'encoder.layers.encoder_layer_6.mlp.3.bias',
                    'blocks.6.mlp.fc2.weight': 'encoder.layers.encoder_layer_6.mlp.3.weight',
                    'blocks.6.attn.proj.bias': 'encoder.layers.encoder_layer_6.self_attention.out_proj.bias',
                    'blocks.6.attn.proj.weight': 'encoder.layers.encoder_layer_6.self_attention.out_proj.weight',
                    'blocks.6.attn.qkv.bias': 'encoder.layers.encoder_layer_6.self_attention.in_proj_bias',
                    'blocks.6.attn.qkv.weight': 'encoder.layers.encoder_layer_6.self_attention.in_proj_weight',
                    'blocks.7.norm1.bias': 'encoder.layers.encoder_layer_7.ln_1.bias',
                    'blocks.7.norm1.weight': 'encoder.layers.encoder_layer_7.ln_1.weight',
                    'blocks.7.norm2.bias': 'encoder.layers.encoder_layer_7.ln_2.bias',
                    'blocks.7.norm2.weight': 'encoder.layers.encoder_layer_7.ln_2.weight',
                    'blocks.7.mlp.fc1.bias': 'encoder.layers.encoder_layer_7.mlp.0.bias',
                    'blocks.7.mlp.fc1.weight': 'encoder.layers.encoder_layer_7.mlp.0.weight',
                    'blocks.7.mlp.fc2.bias': 'encoder.layers.encoder_layer_7.mlp.3.bias',
                    'blocks.7.mlp.fc2.weight': 'encoder.layers.encoder_layer_7.mlp.3.weight',
                    'blocks.7.attn.proj.bias': 'encoder.layers.encoder_layer_7.self_attention.out_proj.bias',
                    'blocks.7.attn.proj.weight': 'encoder.layers.encoder_layer_7.self_attention.out_proj.weight',
                    'blocks.7.attn.qkv.bias': 'encoder.layers.encoder_layer_7.self_attention.in_proj_bias',
                    'blocks.7.attn.qkv.weight': 'encoder.layers.encoder_layer_7.self_attention.in_proj_weight',
                    'blocks.8.norm1.bias': 'encoder.layers.encoder_layer_8.ln_1.bias',
                    'blocks.8.norm1.weight': 'encoder.layers.encoder_layer_8.ln_1.weight',
                    'blocks.8.norm2.bias': 'encoder.layers.encoder_layer_8.ln_2.bias',
                    'blocks.8.norm2.weight': 'encoder.layers.encoder_layer_8.ln_2.weight',
                    'blocks.8.mlp.fc1.bias': 'encoder.layers.encoder_layer_8.mlp.0.bias',
                    'blocks.8.mlp.fc1.weight': 'encoder.layers.encoder_layer_8.mlp.0.weight',
                    'blocks.8.mlp.fc2.bias': 'encoder.layers.encoder_layer_8.mlp.3.bias',
                    'blocks.8.mlp.fc2.weight': 'encoder.layers.encoder_layer_8.mlp.3.weight',
                    'blocks.8.attn.proj.bias': 'encoder.layers.encoder_layer_8.self_attention.out_proj.bias',
                    'blocks.8.attn.proj.weight': 'encoder.layers.encoder_layer_8.self_attention.out_proj.weight',
                    'blocks.8.attn.qkv.bias': 'encoder.layers.encoder_layer_8.self_attention.in_proj_bias',
                    'blocks.8.attn.qkv.weight': 'encoder.layers.encoder_layer_8.self_attention.in_proj_weight',
                    'blocks.9.norm1.bias': 'encoder.layers.encoder_layer_9.ln_1.bias',
                    'blocks.9.norm1.weight': 'encoder.layers.encoder_layer_9.ln_1.weight',
                    'blocks.9.norm2.bias': 'encoder.layers.encoder_layer_9.ln_2.bias',
                    'blocks.9.norm2.weight': 'encoder.layers.encoder_layer_9.ln_2.weight',
                    'blocks.9.mlp.fc1.bias': 'encoder.layers.encoder_layer_9.mlp.0.bias',
                    'blocks.9.mlp.fc1.weight': 'encoder.layers.encoder_layer_9.mlp.0.weight',
                    'blocks.9.mlp.fc2.bias': 'encoder.layers.encoder_layer_9.mlp.3.bias',
                    'blocks.9.mlp.fc2.weight': 'encoder.layers.encoder_layer_9.mlp.3.weight',
                    'blocks.9.attn.proj.bias': 'encoder.layers.encoder_layer_9.self_attention.out_proj.bias',
                    'blocks.9.attn.proj.weight': 'encoder.layers.encoder_layer_9.self_attention.out_proj.weight',
                    'blocks.9.attn.qkv.bias': 'encoder.layers.encoder_layer_9.self_attention.in_proj_bias',
                    'blocks.9.attn.qkv.weight': 'encoder.layers.encoder_layer_9.self_attention.in_proj_weight',
                    'blocks.10.norm1.bias': 'encoder.layers.encoder_layer_10.ln_1.bias',
                    'blocks.10.norm1.weight': 'encoder.layers.encoder_layer_10.ln_1.weight',
                    'blocks.10.norm2.bias': 'encoder.layers.encoder_layer_10.ln_2.bias',
                    'blocks.10.norm2.weight': 'encoder.layers.encoder_layer_10.ln_2.weight',
                    'blocks.10.mlp.fc1.bias': 'encoder.layers.encoder_layer_10.mlp.0.bias',
                    'blocks.10.mlp.fc1.weight': 'encoder.layers.encoder_layer_10.mlp.0.weight',
                    'blocks.10.mlp.fc2.bias': 'encoder.layers.encoder_layer_10.mlp.3.bias',
                    'blocks.10.mlp.fc2.weight': 'encoder.layers.encoder_layer_10.mlp.3.weight',
                    'blocks.10.attn.proj.bias': 'encoder.layers.encoder_layer_10.self_attention.out_proj.bias',
                    'blocks.10.attn.proj.weight': 'encoder.layers.encoder_layer_10.self_attention.out_proj.weight',
                    'blocks.10.attn.qkv.bias': 'encoder.layers.encoder_layer_10.self_attention.in_proj_bias',
                    'blocks.10.attn.qkv.weight': 'encoder.layers.encoder_layer_10.self_attention.in_proj_weight',
                    'blocks.11.norm1.bias': 'encoder.layers.encoder_layer_11.ln_1.bias',
                    'blocks.11.norm1.weight': 'encoder.layers.encoder_layer_11.ln_1.weight',
                    'blocks.11.norm2.bias': 'encoder.layers.encoder_layer_11.ln_2.bias',
                    'blocks.11.norm2.weight': 'encoder.layers.encoder_layer_11.ln_2.weight',
                    'blocks.11.mlp.fc1.bias': 'encoder.layers.encoder_layer_11.mlp.0.bias',
                    'blocks.11.mlp.fc1.weight': 'encoder.layers.encoder_layer_11.mlp.0.weight',
                    'blocks.11.mlp.fc2.bias': 'encoder.layers.encoder_layer_11.mlp.3.bias',
                    'blocks.11.mlp.fc2.weight': 'encoder.layers.encoder_layer_11.mlp.3.weight',
                    'blocks.11.attn.proj.bias': 'encoder.layers.encoder_layer_11.self_attention.out_proj.bias',
                    'blocks.11.attn.proj.weight': 'encoder.layers.encoder_layer_11.self_attention.out_proj.weight',
                    'blocks.11.attn.qkv.bias': 'encoder.layers.encoder_layer_11.self_attention.in_proj_bias',
                    'blocks.11.attn.qkv.weight': 'encoder.layers.encoder_layer_11.self_attention.in_proj_weight',
                    'pos_embed': 'encoder.pos_embedding',
                    'patch_embed.proj.bias': 'conv_proj.bias',
                    'patch_embed.proj.weight': 'conv_proj.weight',
                    'head.bias': 'heads.head.bias',
                    'head.weight': 'heads.head.weight'
                    # 'pre_logits.fc.bias': ''
}