import os
os.environ['KMP_WARNINGS'] = 'off'
import time
import math
import argparse
import numpy as np
from thop import profile
from PIL import Image

import torch
import torch.nn as nn
from torchsummary import summary
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from src.model import get_model
from src.dataset import get_dataset
from src.utils import get_logger

from ipdb import set_trace as st
def debug():
    if dist.get_rank() == 0:
        st()
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()

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
        batch_size=data_opt['batch_size'],
        num_workers=data_opt['num_workers'],
        pin_memory=data_opt['pin_memory'],
        shuffle=data_opt['shuffle'], drop_last=True)

    if data_opt['val'] is not None:
        Logger.info("testset is Valid.")
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        testloader = torch.utils.data.DataLoader(
            testset,
            sampler=test_sampler,
            batch_size=data_opt['batch_size'],
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory'],
            shuffle=data_opt['shuffle'], drop_last=True)
    else:
        Logger.info("testset is NONE.")
        testloader = None
        test_sampler = None

    return trainloader, testloader, train_sampler, test_sampler

def print_opts(opt, logger, start=2):
    if isinstance(opt, dict):
        for key, value in opt.items():
            if isinstance(value, dict):
                logger.info(' '*start + str(key))
                print_opts(value, logger, start+4)
            else:
                logger.info(' ' * start + str(key) + ' ' * start + str(value))
    else:
        logger.info(' '*start + str(opt))

def get_visualize_img(img): # img: [B T C H W]
    # mean = torch.tensor([0.485, 0.456, 0.406])
    # std = torch.tensor([0.229, 0.224, 0.225])
    # x = img[:8].detach().cpu() * std[None, None, :, None, None] + \
    #     mean[None, None, :, None, None]
    x = img[:8].detach().cpu()
    show_x = torch.clamp(x, min=0, max=1)
    b, t, c, h, w = show_x.shape
    show_x = show_x.permute((0, 3, 1, 4, 2)).numpy()
    show_x = show_x.reshape((b * h, t * w, c)) * 255.
    show_x = Image.fromarray(show_x.astype(np.uint8)).convert('RGB')
    return show_x

def print_unused_params(model):
    for name, param in model.named_parameters():
        if param.grad is None or torch.all(param.grad==0):
            print(name)

if __name__ == '__main__':
    assert torch.cuda.is_available()

    parser = argparse.ArgumentParser(description="VQVAES")
    parser.add_argument('--opt', default=None, type=str, help="config file path")
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--split', default=None, type=str)
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    opt = get_opt_from_yaml(args.opt)

    print("Start to init torch distribution...")
    dist.init_process_group(backend='nccl', init_method='env://')
    print("Finish initializing torch distribution...")

    # get current experiment path
    setup_seed(10)
    writer = None
    Logger = get_logger(logging_file="tmp.log", name="tmp", isopen=True)

    # load train data and test data
    if args.split == 'train':
        opt['dataset']['batch_size'] = 1
        opt['dataset']['step'] = 1
        opt['dataset']['name'] += '_EXT'
        trainloader, _, _, _ = get_data_loaders(opt)
        Logger.info(f"Start to convert training set...")
        from tqdm import tqdm
        td = tqdm(range(len(trainloader)))
        trainloader_iter = enumerate(trainloader)
        for _ in td:
            i, inputs = next(trainloader_iter)
            np.save(os.path.join(args.save_dir, "{:010d}.npy".format(i)), inputs)

    elif args.split == 'test':
        opt['dataset']['batch_size'] = 1
        opt['dataset']['step'] = 20
        opt['dataset']['name'] += '_EXT'
        _, testloader, _, _ = get_data_loaders(opt)
        Logger.info(f"Start to convert testing set...")
        from tqdm import tqdm
        td = tqdm(range(len(testloader)))
        testloader_iter = enumerate(testloader)
        for _ in td:
            i, inputs = next(testloader_iter)
            np.save(os.path.join(args.save_dir, "{:010d}.npy".format(i)), inputs)

    else:
        raise NotImplementedError