
from models.InstanceVPD import Model_InstanceVPD

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(args):
    if args.model == 'InstanceVPD':
        model = Model_InstanceVPD(args)
    else:
        raise NotImplementedError()
    
    return model.to(device)