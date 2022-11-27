import os
os.environ['KMP_WARNINGS'] = 'off'
import time
import math
import argparse
import numpy as np
from thop import profile
from PIL import Image
from tqdm import tqdm
from ipdb import set_trace as st
from einops import rearrange, repeat

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQVAES")
    parser.add_argument('--opt', default=None, type=str, help="config file path")
    parser.add_argument('--ckpt', default=None, type=str, help="ckpt file path")
    parser.add_argument('--seperate', action="store_true")
    parser.add_argument('--tokens_dir', default=None, type=str, required=True, help="")
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    opt = get_opt_from_yaml(args.opt)
    assert os.path.exists(args.tokens_dir)

    assert opt['model']['checkpoint_path'] or args.ckpt
    if args.ckpt is not None:
        opt['model']['checkpoint_path'] = args.ckpt

    train_opt = opt['train']
    train_opt['local_rank'] = args.local_rank
    train_opt['device'] = torch.device('cuda', train_opt['local_rank'])

    dist.init_process_group(backend='nccl', init_method='env://')

    # get current experiment path
    setup_seed(10)
    if dist.get_rank() == 0:
        train_opt['exp_name'] = os.path.basename(os.path.dirname(opt['model']['checkpoint_path']))
        train_opt['save_path'] = os.path.abspath(os.path.dirname(opt['model']['checkpoint_path']))
        assert os.path.exists(train_opt['save_path']), f"{train_opt['save_path']} does not exists!"

        # get experment info dir
        Logger = get_logger(os.path.join(train_opt['save_path'],
                                         'visualize_tokens_{}.log'.format(os.path.basename(opt['model']['checkpoint_path']))),
                            train_opt['exp_name'], True)
    else:
        Logger = get_logger(logging_file=None, name=None, isopen=False)

    # initialize model, optimizer and writer
    model, start_step = get_model(opt)
    model = model.to(train_opt['device'])
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[train_opt['local_rank']],
                                                output_device=train_opt['local_rank'])
    model.eval()

    # visualize tokens
    save_dir = args.tokens_dir + '_visualize'
    os.makedirs(save_dir, exist_ok=True)
    video_files = sorted(os.listdir(args.tokens_dir))
    for file in tqdm(video_files):
        ctoken = np.load(os.path.join(args.tokens_dir, file), allow_pickle=True)

        bg_tokens = rearrange(ctoken.item()['bg_tokens'], '(H W) -> H W', H=32, W=32)
        id_tokens = rearrange(ctoken.item()['id_tokens'], '(H W) -> H W', H=16, W=16)
        mo_tokens = rearrange(ctoken.item()['mo_tokens'], '(T H W) -> T H W', H=8, W=8)
        bg_tokens = repeat(bg_tokens, 'H W -> B T H W', B=1, T=1)
        id_tokens = repeat(id_tokens, 'H W -> B T H W', B=1, T=1)
        mo_tokens = repeat(mo_tokens, 'T H W -> B T H W', B=1)


        bg_tokens = torch.from_numpy(bg_tokens.astype(np.int64)).to(train_opt['device'])
        id_tokens = torch.from_numpy(id_tokens.astype(np.int64)).to(train_opt['device'])
        mo_tokens = torch.from_numpy(mo_tokens.astype(np.int64)).to(train_opt['device'])

        xrec = model.module._decode(bg_tokens, id_tokens, mo_tokens)
        show_xrec = get_visualize_img(xrec)

        if args.separate is False:
            show_xrec.save(os.path.join(save_dir, file.replace('.npy', '.png')))
        else:
            save = os.path.join(save_dir)



