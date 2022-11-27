import os
import time
import math
import wandb
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist

import deepspeed
from tqdm import tqdm

from src.utils import  *
from src.model import get_model
from src.dataset import get_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MoCo")
    parser.add_argument('--opt', required=True, type=str,
                        help="path of the config file")
    parser.add_argument('--mode', type=str, default="test",
                        choices=["test", "sample"])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--cut_prob', type=float)
    parser.add_argument('--not_show_process', action="store_true")
    parser.add_argument('--src_tokens_dir', type=str, default=None)
    parser.add_argument('--src_tokens_file', type=str, default=None)
    parser.add_argument('--batchsize', type=int, default=None)
    parser.add_argument('--timesteps', type=int, default=None)
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
    opt.generation.batch_size = args.batchsize or opt.generation.batch_size
    opt.generation.timesteps = args.timesteps or opt.generation.timesteps
    opt.generation.show_process = (not args.not_show_process) and opt.generation.show_process
    assert not (args.src_tokens_dir and args.src_tokens_file) # deal with single file or all tokens in the dir

    # config of exp_name and exp_path
    opt.model.checkpoint_path = args.ckpt
    assert os.path.exists(opt.model.checkpoint_path)
    opt.train.exp_name = os.path.basename(opt.model.checkpoint_path.split('ckpt')[0])
    opt.train.exp_path = os.path.abspath(opt.model.checkpoint_path.split('ckpt')[0])

    # initialize distributed config
    deepspeed.init_distributed()
    setup_seed(args.seed)

    if dist.get_rank() == 0:
        # set logger
        logger = get_logger(os.path.join(opt.train.exp_path, 'generate.log'),
                            opt.train.exp_name, isopen=True)
        # print config infos
        logger.info(f"** Load OPT file: {args.opt}")
        logger.info(f"** Experiment information is saved in {opt.train.exp_path}")
        logger.info(f"** Prepare to generate...")
    else:
        logger = get_logger(logging_file=None, name=None, isopen=False)

    # load model
    model, start_step, start_epoch = get_model(opt, load_vqvae=True)

    if args.src_tokens_file is not None:
        assert args.mode == "test"
        print(f"***  generate timestep: {model.gen_opt.timesteps}")
        src_tokens = args.src_tokens_file

        tokens = np.load(src_tokens, allow_pickle=True).item()
        bg_tokens = torch.from_numpy(tokens['bg_tokens']).unsqueeze(0)
        id_tokens = torch.from_numpy(tokens['id_tokens']).unsqueeze(0)
        mo_tokens = torch.from_numpy(tokens['mo_tokens'].flatten()).unsqueeze(0)

        s = time.time()
        name = f'TimeStep{opt.generation.timesteps}-STEP{start_step}-CUTP{args.cut_prob}-{time.strftime("%H%M%S", time.localtime())}'
        samples = model.refine(bg_tokens=bg_tokens, id_tokens=id_tokens, mo_tokens=mo_tokens,
                               cut_prob=args.cut_prob, timesteps=opt.generation.timesteps,
                               mode='test', save_path=os.path.join(opt.train.exp_path, 'refine', f'{name}.png'))
        print(f' {name} - {time.time() - s}')

    elif args.src_tokens_dir is not None:
        assert args.mode == "sample"
        # refine tokens in a directory
        print(f"***  cut probability: {args.cut_prob}")
        print(f"***  generate timestep: {model.gen_opt.timesteps}")
        for cur_file in tqdm(os.listdir(args.src_tokens_dir)):
            save_name = cur_file.split('.')[0]
            base_name = os.path.basename(args.src_tokens_dir)
            src_tokens = os.path.join(args.src_tokens_dir, cur_file)

            tokens = np.load(src_tokens, allow_pickle=True).item()
            bg_tokens = torch.from_numpy(tokens['bg_tokens']).to(torch.int32).unsqueeze(0)
            id_tokens = torch.from_numpy(tokens['id_tokens']).to(torch.int32).unsqueeze(0)
            mo_tokens = torch.from_numpy(tokens['mo_tokens'].flatten()).to(torch.int32).unsqueeze(0)

            # samples, tokens = model.refine(bg_tokens=bg_tokens, id_tokens=id_tokens, mo_tokens=mo_tokens,
            #                                cut_prob=args.cut_prob, timesteps=opt.generation.timesteps, mode='sample')
            samples, tokens = model.refine_twosteps(bg_tokens=bg_tokens, id_tokens=id_tokens, mo_tokens=mo_tokens,
                                                    timesteps=opt.generation.timesteps, mode='sample')
            for b in range(model.gen_opt.batch_size):
                # save generated video samples by frame
                # sample_dir = os.path.join(opt.train.exp_path, 'refine_videos',
                #                           f'CKPT{model.start_step}-T{opt.generation.timesteps}-'
                #                           f'from-{base_name}', save_name)
                sample_dir = os.path.join(opt.train.exp_path, 'refine_videos',
                                          f'CKPT{model.start_step}-T{opt.generation.timesteps}-2STAGE-'
                                          f'from-{base_name}', save_name)
                os.makedirs(sample_dir, exist_ok=True)
                for j, cur_x in enumerate(samples[b]):
                    cur_x.save(os.path.join(sample_dir, "frame_{:03d}.png").format(j))

    else:
        raise NotImplementedError











