
from utils.args import *
from torch import nn
from models.utils import get_backbone
import torch
from train.utils import device
from models.tokenPrompt import get_token_prompt

import math
from functools import reduce
from operator import mul
from torch.utils.checkpoint import checkpoint

from models.utils import SimamModule



def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='InsVP')
    add_management_args(parser)
    add_experiment_args(parser)
    
    
    
class Model_InstanceVPD(nn.Module):
    def __init__(self, args):
        super(Model_InstanceVPD, self).__init__()
        self.args = args
        self.backbone = get_backbone(args)
        self.mode = "train"
        if self.args.simam == "True":
            # print("utilize simam")
            self.simam = SimamModule()    
        
        self.prompts = nn.Parameter(torch.randn(args.deep_layer, args.p_len_vpt, 768))
        self.prompts.data.uniform_(-1, 1)
        self.prompt_dropout = torch.nn.Dropout(self.args.prompt_dropout)
        self.TokenPrompt = get_token_prompt(args)
            
        self.meta_dropout = torch.nn.Dropout(0.1)
        self.meta_dropout_2 = torch.nn.Dropout(0.1)
        
        self.prompt_patch = args.prompt_patch
        n = self.prompt_patch
        h = args.hid_dim
        self.meta_net = nn.Sequential(
            nn.Linear(3*n*n, h),
            nn.ReLU(),
            nn.Linear(h, 3*n*n)
        )   
        self.prompt_patch_2 = args.prompt_patch_2
        self.prompt_patch_22 = args.prompt_patch_22
        n_2 = self.prompt_patch_2
        n_22 = self.prompt_patch_22
        self.meta_net_2 = nn.Sequential(
            nn.Conv2d(3, args.hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
            nn.ReLU(),
            nn.Conv2d(args.hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
        )
        
    
    def train(self):
        self.backbone.eval()
        self.backbone.head.train()
        self.meta_net.train()
        self.meta_net_2.train()
        self.meta_dropout.train()
        self.meta_dropout_2.train()
        self.prompt_dropout.train()
        self.TokenPrompt.train()
    
    def eval(self):
        self.backbone.eval()
        self.backbone.head.eval()
        self.meta_net.eval()
        self.meta_net_2.eval()
        self.meta_dropout.eval()
        self.meta_dropout_2.eval()
        self.prompt_dropout.eval()
        self.TokenPrompt.eval()
                         
    
    def forward(self, x):
        if self.args.simam == "True":
            x = self.simam(x)
        prompts = self.get_prompts(x)  
        x = x + prompts     
        x = self.forward_deep_VPD(x)
        return x  
    
    def forward_deep_VPD(self, x, get_feature=False):
        ori_image = x
        bk = self.backbone
        x = bk.patch_embed(x)
        x = bk._pos_embed(x)
        x = bk.norm_pre(x)
        x = self.forward_block(x, ori_image)
        x = bk.norm(x)  
        x = x[:, 0]
        if get_feature:
            return bk.head(bk.fc_norm(x)), x
        x = bk.fc_norm(x)
        return bk.head(x)
    
    def forward_block(self, x, ori_image):
        bk = self.backbone
        for i, block in enumerate(bk.blocks):
            B = x.shape[0]       
            if i < self.args.deep_layer:
                p_len = self.args.p_len_vpt
                prompt = self.prompt_dropout(self.prompts[i].expand(B, -1, -1)) # [p_len, 768]
                prompt = prompt + self.TokenPrompt(ori_image, layer=i)
                if i == 0:
                    x = torch.cat((x[:, :1, :], prompt, x[:, 1:, :]), dim=1)
                else:
                    x = torch.cat((x[:, :1, :], prompt, x[:, (1+p_len):, :]), dim=1)
            x = block(x)
        return x
    
    
    def get_local_prompts(self, x):
        # [64, 3, 224, 224]
        B = x.shape[0]
        n = self.prompt_patch
        n_patch = int(224 / n)
        x = x.reshape(B, 3, n_patch, n, n_patch, n) # [64, 3, 14, 16, 14, 16]
        x = x.permute(0, 2, 4, 1, 3, 5) # [64, 14, 14, 3, 16, 16]
        x = x.reshape(B, n_patch*n_patch, 3*n*n)
        x = x.reshape(B*n_patch*n_patch, 3*n*n)
        x = self.meta_net(x)
        x = x.reshape(B, n_patch, n_patch, 3, n, n)
        x = x.permute(0, 3, 1, 4, 2, 5) # [64, 3, 14, 16, 14, 16]
        x = x.reshape(B, 3, 224, 224)
        return self.meta_dropout(x)
    
    def get_prompts(self, x):
        prompts_1 = self.get_local_prompts(x)
        x = self.meta_dropout_2(self.meta_net_2(x))
        return prompts_1 + self.args.prompts_2_weight * x
    
    def get_classifier(self):
        if self.args.arch == 'ViT/B-16':
            if self.args.pretrained == 'imagenet22k':
                classifier = self.backbone.head
            else:   
                classifier = self.backbone.heads
        elif self.args.arch in ['resnet50', 'resnet18']:
            classifier = self.backbone.fc
        return classifier
    
    def learnable_parameters(self):
        if self.args.arch in ['ViT/B-16', 'swin']:
            if self.args.pretrained == 'imagenet22k':
                params = list(self.backbone.head.parameters())
            else:   
                params = list(self.backbone.heads.parameters())
        elif self.args.arch in ['resnet50', 'resnet18']:
            params = list(self.backbone.fc.parameters())
            
        params += list(self.meta_net.parameters())
        params += list(self.meta_dropout.parameters())
        params += list(self.meta_dropout_2.parameters())
        if self.args.deep_prompt_type in ["ours9"]:
            params += [self.prompts]
            params += list(self.prompt_dropout.parameters())
        elif self.args.deep_prompt_type in ["ours9"]:
            params += list(self.TokenPrompt.parameters())
        params += list(self.meta_net_2.parameters())
        return params 
    
