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
    parser.add_argument('--generator', type=str, default=None)
    parser.add_argument('--num_samples', type=int)
    # parser.add_argument('--noise_weight', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=0)
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
    opt.generation.temperature = args.temperature
    # opt.generation.noise_weight = args.noise_weight
    opt.generation.name = args.generator or opt.generation.get('name', 'default')
    opt.generation.batch_size = args.batchsize or opt.generation.batch_size
    opt.generation.samples = args.num_samples or opt.generation.samples
    opt.generation.timesteps = args.timesteps or opt.generation.timesteps
    opt.generation.show_process = opt.generation.show_process and (args.mode=="test")

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
    print(f"***  generator type: {model.gen_opt.name}")
    print(f"***  generate timestep: {model.gen_opt.timesteps}")
    # print(f"***  generate noise_weight: {opt.generation.noise_weight}")
    print(f"***  generate temperature: {opt.generation.temperature}")
    print(f"***  save exp_path: {opt.train.exp_path}")

    if args.mode == "test":
        s = time.time()
        name = f'TimeStep{opt.generation.timesteps}-STEP{start_step}'
        samples = model.generate(mode=args.mode,
                                 save_path=os.path.join(opt.train.exp_path, 'gen', f'{name}.png'))
        print(f' {name} - {time.time() - s}')
    elif args.mode == "sample":
        base_name = f'CKPT{model.start_step}-' \
                    f'T{opt.generation.timesteps}-' \
                    f'TYPE{model.gen_opt.name}-' \
                    f'TEMP{model.gen_opt.temperature}-2STAGE'
        # base_name = f'CKPT{model.start_step}-' \
        #             f'T{opt.generation.timesteps}-' \
        #             f'TEMP{model.gen_opt.temperature}-2STAGE-GUB'
        print(f"***  save DIR: {base_name}")

        for i in tqdm(range(model.gen_opt.samples // model.gen_opt.batch_size)):
            # samples, tokens = model.generate(mode=args.mode)
            samples, tokens = model.generate_twosteps(mode=args.mode)
            for b in range(model.gen_opt.batch_size):
                # save generated video samples by frame
                sample_dir = os.path.join(opt.train.exp_path, 'sample_videos',
                                          base_name, f'sample{i}-batch{b}-seed{args.seed}')
                os.makedirs(sample_dir, exist_ok=True)
                for j, cur_x in enumerate(samples[b]):
                    cur_x.save(os.path.join(sample_dir, "frame_{:03d}.png").format(j))
                # save generated video tokens
                token_dir = os.path.join(opt.train.exp_path, 'sample_tokens', base_name)
                os.makedirs(token_dir, exist_ok=True)
                np.save(os.path.join(token_dir, f'sample{i}-batch{b}-seed{args.seed}.npy'), tokens[b])
    else:
        raise NotImplementedError










