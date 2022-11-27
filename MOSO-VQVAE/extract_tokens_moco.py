import os
os.environ['KMP_WARNINGS'] = 'off'
import time
import math
import argparse
import numpy as np
from thop import profile
from PIL import Image
from tqdm import tqdm
from einops import rearrange

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
    print(f"Start to get dataset {data_opt['name']}...")
    trainset, testset = get_dataset(opt)

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=data_opt['batch_size'],
        num_workers=data_opt['num_workers'],
        pin_memory=True,
        shuffle=False, drop_last=True)

    test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
    testloader = torch.utils.data.DataLoader(
        testset,
        sampler=test_sampler,
        batch_size=data_opt['batch_size'],
        num_workers=data_opt['num_workers'],
        pin_memory=True,
        shuffle=False, drop_last=True)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQVAES")
    parser.add_argument('--opt', default=None, type=str, help="config file path")
    parser.add_argument('--ckpt', default=None, type=str, help="ckpt file path")
    parser.add_argument('--save_dir', default=None, type=str, required=True, help="")
    parser.add_argument('--cur_fps', default=None, type=int, help="")
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_valid', action='store_true')
    parser.add_argument('--onlyfirstframe', action='store_true')
    parser.add_argument('--firsttwoframes', action='store_true')
    parser.add_argument('--firstkframe_k', type=int, default=-1)
    parser.add_argument('--aug_trainset', action='store_true')
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    opt = get_opt_from_yaml(args.opt)
    assert os.path.exists(args.save_dir)

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
                                         'extract_tokens_{}.log'.format(os.path.basename(opt['model']['checkpoint_path']))),
                            train_opt['exp_name'], True)
    else:
        Logger = get_logger(logging_file=None, name=None, isopen=False)

    # load train data and test data
    opt['dataset']['name'] += '_EXT'
    if args.onlyfirstframe:
        opt['dataset']['name'] += '_FirstFrame'
    if args.firsttwoframes:
        opt['dataset']['name'] += '_FirstTwoFrames'
    if args.firstkframe_k != -1:
        opt['dataset']['K'] = args.firstkframe_k
        opt['dataset']['name'] += '_FirstKFrames'
        if args.aug_trainset:
            opt['dataset']['step'] = 1
    trainloader, testloader, trainsampler, testsampler = get_data_loaders(opt)

    # initialize model, optimizer and writer
    model, start_step = get_model(opt)
    model = model.to(train_opt['device'])
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[train_opt['local_rank']],
                                                output_device=train_opt['local_rank'])
    model.eval()

    # start iteration
    val_time = time.time()
    Logger = get_logger()
    testloader_iter = enumerate(testloader)
    trainloader_iter = enumerate(trainloader)

    if not args.skip_train:
        # extract trainset tokens
        tot_iter = len(trainloader)
        for step in tqdm(range(tot_iter)):
            _, inputs = next(trainloader_iter)
            with torch.no_grad():
                x = [inputs['ret_img'], inputs['ret_img'], inputs['ret_img'], inputs['ret_img_mo']]  # Full BG ID
                x = [item.to(train_opt['device']) for item in x]
                output = model.module._generater(x, is_training=False)

            video_ids, starts = inputs['video_id'], inputs['start']
            for i, cvideo in enumerate(video_ids):
                cstart = starts[i]

                bg_tokens = output['vq_output_bg']['encoding_indices'][i].detach().cpu().numpy()
                id_tokens = output['vq_output_id']['encoding_indices'][i].detach().cpu().numpy()
                mo_tokens = output['vq_output_mo']['encoding_indices'].detach().cpu().numpy()
                mo_tokens = rearrange(mo_tokens, '(b t) h w -> b t h w', b=x[0].shape[0])[i]

                if args.cur_fps is None:
                    save_dir = os.path.join(args.save_dir, 'train', cvideo)
                else:
                    save_dir = os.path.join(args.save_dir, 'train', f'FPS{args.cur_fps}', cvideo)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"start{cstart}_rank{dist.get_rank()}.npy")
                save_arr = {'bg_tokens': bg_tokens, 'id_tokens': id_tokens, 'mo_tokens': mo_tokens}
                np.save(save_path, save_arr)

    if not args.skip_valid:
        # extract testset tokens
        tot_iter = len(testloader)
        for step in tqdm(range(tot_iter)):
            _, inputs = next(testloader_iter)
            with torch.no_grad():
                x = [inputs['ret_img'], inputs['ret_img'], inputs['ret_img'], inputs['ret_img_mo']]  # Full BG ID
                x = [item.to(train_opt['device']) for item in x]
                output = model.module._generater(x, is_training=False)

            video_ids, starts = inputs['video_id'], inputs['start']
            for i, cvideo in enumerate(video_ids):
                cstart = starts[i]

                bg_tokens = output['vq_output_bg']['encoding_indices'][i].detach().cpu().numpy()
                id_tokens = output['vq_output_id']['encoding_indices'][i].detach().cpu().numpy()
                mo_tokens = output['vq_output_mo']['encoding_indices'].detach().cpu().numpy()
                mo_tokens = rearrange(mo_tokens, '(b t) h w -> b t h w', b=x[0].shape[0])[i]

                if args.cur_fps is None:
                    save_dir = os.path.join(args.save_dir, 'valid', cvideo) # TO RECOVER
                    # save_dir = os.path.join(args.save_dir, cvideo)
                else:
                    save_dir = os.path.join(args.save_dir, 'valid', f'FPS{args.cur_fps}', cvideo)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"start{cstart}_rank{dist.get_rank()}.npy")
                save_arr = {'bg_tokens': bg_tokens, 'id_tokens': id_tokens, 'mo_tokens': mo_tokens}
                np.save(save_path, save_arr)


