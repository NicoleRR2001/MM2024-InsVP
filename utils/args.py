# Desc: Argument parser for the project

from argparse import ArgumentParser
import time
import os


# modularized arguments management
def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--mode', type=str, 
                        default="train",
                        help='train mode or eval mode')
    parser.add_argument('--debug',
                        default=False,
                        action='store_true',
                        help='debug mode')
    parser.add_argument('--dataset', type=str, 
                        # required=True,
                        default="cifar100",
                        # choices=['cifar10', 'cifar100', 'imagenet', 'flower102', 
                        #          'food101', 'FGVCAircraft', 'EuroSAT', 'OxfordIIITPet', 
                        #          'DTD', 'SVHN', 'GTSRB', 'StanfordCars', 'StanfordDogs'],
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--arch', type=str, 
                        default="ViT/B-16",
                        help='The architecture of backbone.')
    parser.add_argument('--pretrained', type=str, 
                        default="imagenet1k",
                        help='The pretrained weights of backbone.')
    parser.add_argument('--n_epochs', type=int, 
                        default=30,
                        help='The training epochs.')
    parser.add_argument('--optimizer', type=str,
                        default="AdamW",
                        help='The optimizer.')
    parser.add_argument('--lr', type=float, 
                        default=3e-5,
                        help='The learning rate.')
    parser.add_argument('--lr_c', type=float, 
                        default=1e-1,
                        help='The learning rate.')
    parser.add_argument('--batch_size', type=int,
                        default=64,
                        help='The batch size.')
    parser.add_argument('--scheduler', type=str,
                        default="none",
                        help='The scheduler.')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0001,
                        help='The weight decay.')
    parser.add_argument('--weight_decay_c', type=float,
                        default=0,
                        help='The weight decay.')
    parser.add_argument('--transform', type=str,
                        default='default',
                        choices=['default', 'SOTA'],
                        help='The transform method.')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument("--bias_multiplier", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--trainer", type=str, default="ours")
    parser.add_argument("--resize_dim", type=int, default=256)
    
    # for tester
    parser.add_argument('--model_load_path', type=str,
                        default='',
                        help='The path of load model.')
    
    parser.add_argument("--mixup", type=str,
                        default="none")
    parser.add_argument("--cutmix_alpha", type=float,
                        default=0.0)
    

    
    
def add_model_args(parser: ArgumentParser) -> None:

    # for InsVP
    parser.add_argument('--meta_net', type=int,
                        default=0,
                        help='The meta net of InstanceVP.')
    parser.add_argument('--prompt_patch', type=int,
                        default=16,
                        help='The prompt patch size of InstanceVP.')
    parser.add_argument("--prompt_patch_12", type=int,
                        default=31,
                        help="The prompt patch size of InstanceVP.")
    parser.add_argument("--prompt_patch_2", type=int,
                        default=21,
                        help="The prompt patch size of InstanceVP.")
    parser.add_argument("--prompt_patch_22", type=int,
                        default=31,
                        help="The prompt patch size of InstanceVP.")
    parser.add_argument("--prompt_patch_3", type=int,
                        default=64,
                        help="The prompt patch size of InstanceVP.")
    parser.add_argument('--hid_dim', type=int,
                        default=64,
                        help='The hidden dimension of InstanceVP.')
    parser.add_argument("--hid_dim_2", type=int,
                        default=6,
                        help="The hidden dimension of InstanceVP.")
    parser.add_argument("--global_prompts_weight", type=float,
                        default=1,
                        help="The weight of global prompts.")
    parser.add_argument("--prompts_2_weight", type=float,
                        default=1,
                        help="The weight of prompts 2.")
    parser.add_argument("--prompts_3_weight", type=float,
                        default=1,
                        help="The weight of prompts 3.")
    parser.add_argument("--nheads", type=int,
                        default=1,
                        help="The number of heads in multi-head attention.")
    parser.add_argument("--deep_layer", type=int,
                        default=12,
                        help="The number of deep layer.")
    parser.add_argument("--p_len_vpt", type=int,
                        default=10,
                        help="The length of prompt.")
    parser.add_argument("--p_len", type=int, default=10,
                        help="The length of prompt.")
    parser.add_argument("--deep_prompt_type", type=str,
                        default='vpt')
    parser.add_argument("--instance_prompt", type=str,
                        default="instance_prompt")
    parser.add_argument("--prompt_dropout", type=float,
                        default=0.1)
    parser.add_argument("--meta_bn", type=str, default="none")
    parser.add_argument("--TP_kernel_1", type=int, default=3)
    parser.add_argument("--TP_kernel_2", type=int, default=3)
    parser.add_argument("--TP_kernel_3", type=int, default=3)
    parser.add_argument("--token_prompt_eta", type=float, default=0.5)
    parser.add_argument("--token_prompt_type", type=str, default="add")
    parser.add_argument("--cls", type=str, default="default")
    parser.add_argument("--simam", type=str, default="False")
    

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--info', type=str, 
                        default="Debug",
                        help='The information of the experiment.')
    parser.add_argument('--seed', type=int, 
                        default="3407",
                        help='set the random seed')
    parser.add_argument('--output_path', type=str,
                        default='./Output/Debug',
                        help='The path to save the output.')
    parser.add_argument('--loader', type=str,
                        default='DAM-VP')
    parser.add_argument('--warm_prompt', type=bool,
                        default=False)
    
    
    parser.add_argument('--base_dir', type=str, default='')
    parser.add_argument('--dataset_perc', default=1.0, type=float, help='Dataset percentage for usage [default: 1.0].')
    parser.add_argument('--crop_size', default=224, type=int, help='Input size of images [default: 224].')
    parser.add_argument('--pretrained_model', type=str, default='vit-b-22k')
    parser.add_argument('--distributed', action='store_true', default=False, help='Whether to use the distributed mode [default: False].')
    parser.add_argument('--num_workers', type=int, default=4, help='Worker nums of data loading.')
    parser.add_argument('--pin_memory', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--num_gpus', type=int, default=1, help='Num of GPUs to use.')
    
    parser.add_argument('--NUM_GPUS', type=int, default=1, help='Num of GPUs to use.')
    parser.add_argument('--data_dir', type=str, default='')
    
    
      

def save_args(args):
    """
    Save the arguments into a txt file.
    """
    output_path = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '-' + args.info
    output_path = os.path.join(args.output_path, output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    import json
    with open(os.path.join(output_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    args.output_path = output_path
    return args
    