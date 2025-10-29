import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

import functools
from einops import rearrange

from models.scratch_former import ScratchFormer

###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            return 1.0 - epoch / float(args.max_epochs + 1)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs // 3
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented' % args.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print(f"[Init] Initializing network with {init_type}")
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize network with device (GPU or CPU), and optional multi-GPU support."""
    use_cuda = torch.cuda.is_available() and len(gpu_ids) > 0
    device = torch.device(f'cuda:{gpu_ids[0]}' if use_cuda else 'cpu')

    print(f"[Device] Using device: {device}")
    net.to(device)

    if use_cuda and len(gpu_ids) > 1:
        net = torch.nn.DataParallel(net, gpu_ids)

    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'ScratchFormer':
        net = ScratchFormer(embed_dim=args.embed_dim)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)

    return init_net(net, init_type, init_gain, gpu_ids)
