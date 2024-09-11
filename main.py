# Author: Zichen Liu

import importlib
import torch

from argparse import ArgumentParser
from utils.seed import set_random_seed
from utils.args import add_experiment_args, add_management_args, add_model_args, save_args

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "9"

def main():


    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    parser = ArgumentParser(description='perturbation', allow_abbrev=False)
    # print("main.py")
    parser.add_argument('--model', type=str, 
                        # required=True,
                        default='perturbation',
                        help='Model name.', choices=['InstanceVPD'])
    
    args = parser.parse_known_args()[0]
    
    # add arguments for the specific model
    # mod = importlib.import_module('models.' + args.model)
    
    # get_parser = getattr(mod, 'get_parser')
    # parser = get_parser()
    add_management_args(parser)
    add_experiment_args(parser)
    add_model_args(parser)
    args = parser.parse_args()
    
    if args.seed is not None:
        print("Setting random seed to {}".format(args.seed))
        set_random_seed(args.seed)
        
    args = save_args(args)   
    
    if args.trainer == "ours":  
        from train import trainer
        
    # print(args.mode)
    # input()
         
    if args.mode == 'train':
        trainer.train(args)
   
if __name__ == '__main__':
    # print("main.py")
    main()
