import os
import time
import math
import json
import wandb
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data as Data

import deepspeed
from tqdm import tqdm

from src.utils import  *
from src.model import get_model
from src.dataset.CustomVPDataset import CustomVPDataset
from src.dataset.CustomVPDataset_FixPreMO import CustomVPDataset_FixPreMo


def get_dataloader_custom(opt, split, items):
    if split == 'test':
        validset = CustomVPDataset(tokens_1th_dir=opt.dataset.valid.tokens_1th_dir,
                                   tokens_16f_dir=opt.dataset.valid.tokens_16f_dir,
                                   items=items)
        valid_sampler = Data.distributed.DistributedSampler(validset)
        validloader = Data.DataLoader(
            validset,
            sampler=valid_sampler,
            batch_size=opt.train.batch_size,
            pin_memory=opt.dataset.pin_memory,
            shuffle=opt.dataset.shuffle,
            drop_last=True,
        )
        return validloader, valid_sampler
    elif split == 'train':
        trainset = CustomVPDataset(tokens_1th_dir=opt.dataset.train.tokens_1th_dir,
                                   tokens_16f_dir=opt.dataset.train.tokens_16f_dir,
                                   items=items)
        train_sampler = Data.distributed.DistributedSampler(trainset)
        trainloader = Data.DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=opt.train.batch_size,
            pin_memory=opt.dataset.pin_memory,
            shuffle=opt.dataset.shuffle,
            drop_last=True
        )
        return trainloader, train_sampler
    else:
        raise NotImplementedError

def get_dataloader_FixPreMo(opt, split, items):
    if split == 'test':
        validset = CustomVPDataset_FixPreMo(tokens_pre_dir=opt.dataset.valid.tokens_pre_dir,
                                            tokens_tar_dir=opt.dataset.valid.tokens_tar_dir,
                                            pre_frame_num=opt.dataset.pre_frame_num,
                                            items=items)
        valid_sampler = Data.distributed.DistributedSampler(validset)
        validloader = Data.DataLoader(
            validset,
            sampler=valid_sampler,
            batch_size=opt.train.batch_size,
            pin_memory=opt.dataset.pin_memory,
            shuffle=opt.dataset.shuffle,
            drop_last=True,
        )
        return validloader, valid_sampler
    elif split == 'train':
        trainset = CustomVPDataset_FixPreMo(tokens_pre_dir=opt.dataset.train.tokens_pre_dir,
                                            tokens_tar_dir=opt.dataset.train.tokens_tar_dir,
                                            pre_frame_num=opt.dataset.pre_frame_num,
                                            items=items)
        train_sampler = Data.distributed.DistributedSampler(trainset)
        trainloader = Data.DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=opt.train.batch_size,
            pin_memory=opt.dataset.pin_memory,
            shuffle=opt.dataset.shuffle,
            drop_last=True
        )
        return trainloader, train_sampler
    else:
        raise NotImplementedError



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MoCo")
    parser.add_argument('--opt', required=True, type=str,
                        help="path of the config file")
    parser.add_argument('--split', type=str, default="test", choices=["train", "test"],
                        help="Give visualized tokens from train/test dataset")
    parser.add_argument('--seed', type=int, help="random seed")
    parser.add_argument('--num_samples', type=int,
                        help="number of total predict items")
    parser.add_argument('--temperature', type=float, default=0,
                        help="Degree of sample randomness")
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--items_json', type=str,
                        help="path to dataset items json file")
    parser.add_argument('--timesteps', type=int, default=None,
                        help="Total timesteps of predicting a sample")
    parser.add_argument('--ckpt', required=True, type=str)
    parser.add_argument('--local_rank', type=int)
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    # global config
    opt = get_obj_from_yaml(args.opt)
    opt.train.args = args
    opt.train.local_rank = args.local_rank
    opt.train.device = torch.device('cuda', args.local_rank)
    opt.generation.temperature = args.temperature
    opt.train.batch_size = args.batchsize or opt.generation.batch_size
    opt.generation.batch_size = args.batchsize or opt.generation.batch_size
    opt.generation.samples = args.num_samples or opt.generation.samples
    opt.generation.timesteps = args.timesteps or opt.generation.timesteps

    # config of exp_name and exp_path
    opt.model.checkpoint_path = args.ckpt
    assert os.path.exists(opt.model.checkpoint_path)
    opt.train.exp_name = os.path.basename(opt.model.checkpoint_path.split('ckpt')[0])
    opt.train.exp_path = os.path.abspath(opt.model.checkpoint_path.split('ckpt')[0])

    # initialize distributed config
    deepspeed.init_distributed()
    setup_seed(args.seed)

    # set logger
    if dist.get_rank() == 0:
        logger = get_logger(os.path.join(opt.train.exp_path, 'generate.log'),
                            opt.train.exp_name, isopen=True)
        logger.info(f"** Load OPT file: {args.opt}")
        logger.info(f"** Experiment information is saved in {opt.train.exp_path}")
        logger.info(f"** Prepare to generate...")
    else:
        logger = get_logger(logging_file=None, name=None, isopen=False)

    # load model
    model, start_step, start_epoch = get_model(opt, load_vqvae=True)
    print(f"***  generate timestep: {model.gen_opt.timesteps}")
    print(f"***  generate temperature: {model.gen_opt.temperature}")
    print(f"***  save exp_path: {opt.train.exp_path}")

    # load data for the first sample
    items = json.load(open(args.items_json, 'r'))
    if opt.dataset.cname == 'CustomVPDataset_FixPreMo':
        get_dataloader = get_dataloader_FixPreMo
    else:
        get_dataloader = get_dataloader_custom
    dataloader, _ = get_dataloader(opt, args.split, items)
    dataloader_iter = iter(dataloader)

    # start prediction
    base_name = f'CKPT{model.start_step}-T{opt.generation.timesteps}-' \
                f'TEMP{model.gen_opt.temperature}-{args.split}'
    print(f"***  save DIR: {base_name}")

    st()
    assert model.gen_opt.batch_size == 1
    for i in tqdm(range(model.gen_opt.samples // model.gen_opt.batch_size)):
        inputs = next(dataloader_iter)
        item = inputs['video_item']
        sample_dir = os.path.join(opt.train.exp_path, 'interpolate_videos',
                                  base_name, f'{item[0][0]}-{item[1][0].split(".")[0]}-seed{args.seed}')
        if os.path.exists(sample_dir):
            continue

        # interpolate
        samples, tokens = model.interpolate(inputs=inputs, mode="sample")

        # save
        os.makedirs(sample_dir)
        for j, cur_x in enumerate(samples[0]):
            cur_x.save(os.path.join(sample_dir, "frame_{:03d}.png").format(j))





