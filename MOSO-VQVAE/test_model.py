import os
os.environ['KMP_WARNINGS'] = 'off'
import time
import math
import argparse
import numpy as np
from thop import profile

import torch
import torch.nn as nn
from torchsummary import summary
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from src.model import get_model
from src.dataset import get_dataset
from src.utils import get_logger

# set random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_opt_from_yaml(path):
    assert os.path.exists(path), f"{path} must exists!"
    import yaml
    with open(path, 'r') as f:
        opt = yaml.load(f)
    return opt

def get_data_loaders(opt):
    data_opt = opt['dataset']
    Logger = get_logger()
    Logger.info(f"Start to get dataset {data_opt['name']}...")
    trainset, testset = get_dataset(opt)

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=data_opt['train']['batch_size'],
        num_workers=data_opt['train']['num_workers'],
        pin_memory=data_opt['train']['pin_memory'],
        shuffle=data_opt['train']['shuffle'], drop_last=True)

    if data_opt['val'] is not None:
        Logger.info("testset is Valid.")
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        testloader = torch.utils.data.DataLoader(
            testset,
            sampler=test_sampler,
            batch_size=data_opt['val']['batch_size'],
            num_workers=data_opt['val']['num_workers'],
            pin_memory=data_opt['val']['pin_memory'],
            shuffle=data_opt['val']['shuffle'], drop_last=True)
    else:
        Logger.info("testset is NONE.")
        testloader = None

    return trainloader, testloader

def print_opts(opt, logger, start=0):
    if isinstance(opt, dict):
        for key, value in opt.items():
            logger.info(' '*start + str(key))
            print_opts(value, logger, start+4)
    else:
        logger.info(' '*start + str(opt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQVAES")
    parser.add_argument('--opt', default=None, type=str, help="config file path")
    args = parser.parse_args()

    opt = get_opt_from_yaml(args.opt)
    train_opt = opt['train']

    # get current experiment path
    setup_seed(10)
    Logger = get_logger("", None, False)
    # initialize model, optimizer and writer
    model, start_step = get_model(opt)
    model = model.cuda()

    # test model params
    lq = torch.rand(2, 3, 4, opt['dataset']['train']['lq_img_size'],
                        opt['dataset']['train']['lq_img_size']).cuda()
    gt = torch.rand(2, 3, 4, opt['dataset']['train']['gt_img_size'],
                        opt['dataset']['train']['gt_img_size']).cuda()
    macs, params = profile(model.eval(), inputs=(lq, gt, False))
    print(f'Total macc: {macs}')
    print(f'Total params:{params}')
