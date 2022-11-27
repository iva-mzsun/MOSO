import os
import ipdb
os.environ['KMP_WARNINGS'] = 'off'
import time
import math
import argparse
import numpy as np
from thop import profile
from PIL import Image
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as Data
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

def save_videos(videos_tensor, nrow, path):
    b, c, t, h, w = videos_tensor.shape
    imgs_tensor = videos_tensor.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
    imgs = make_grid(imgs_tensor, nrow=nrow, normalize=True)
    img = F.to_pil_image(imgs.detach())
    show_img = Image.fromarray(np.array(img))
    show_img.save(path)

def get_data_loaders(opt):
    data_opt = opt['dataset']
    Logger = get_logger()
    Logger.info(f"Start to get dataset {data_opt['name']}...")
    trainset, testset = get_dataset(opt)

    train_sampler = Data.distributed.DistributedSampler(trainset)
    trainloader = Data.DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=data_opt['batch_size'],
        num_workers=data_opt['num_workers'],
        pin_memory=data_opt['pin_memory'],
        shuffle=data_opt['shuffle'],
        drop_last=True)

    if data_opt['val'] is not None:
        Logger.info("testset is Valid.")
        test_sampler = Data.distributed.DistributedSampler(testset)
        testloader = Data.DataLoader(
            testset,
            sampler=test_sampler,
            batch_size=data_opt['batch_size'],
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory'],
            shuffle=data_opt['shuffle'],
            drop_last=True)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQVAES")
    parser.add_argument('--opt', default=None, type=str, help="config file path")
    parser.add_argument('--ckpt', default=None, type=str, help="ckpt file path")
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    opt = get_opt_from_yaml(args.opt)

    assert opt['model']['checkpoint_path'] is not None or args.ckpt is not None
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
                                         'valid_{}.log'.format(os.path.basename(opt['model']['checkpoint_path']))),
                            train_opt['exp_name'], True)
    else:
        Logger = get_logger(logging_file=None, name=None, isopen=False)

    # load train data and test data
    trainloader, testloader, trainsampler, testsampler = get_data_loaders(opt)

    # initialize model, optimizer and writer
    model, start_step = get_model(opt)
    model = model.to(train_opt['device'])
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[train_opt['local_rank']],
                                                output_device=train_opt['local_rank'])

    # start iteration
    Logger.info(f"====================validation start========================")
    Logger.info(f"Start to Validation for ckpt {opt['model']['checkpoint_path']}..")
    model.eval()
    stime = time.time()
    testloader_iter = enumerate(testloader)

    rec_loss, ssim_metric, msssim_metric, lpips = [], [], [], []
    tot_iter = len(testloader)
    for val_step in tqdm(range(tot_iter)):
        _, val_inputs = next(testloader_iter)
        with torch.no_grad():
            val_inputs = [item.to(train_opt['device']) for item in val_inputs]
            xfull, xbg, xid, xmo = val_inputs
            val_inputs = [xfull, xfull, xfull, xmo]  # Full BG ID
            output = model(val_inputs, is_training=False, writer=None)
        cur_ssim = output['ssim_metric'].clone().detach().cpu().numpy() if 'ssim_metric' in output.keys() else 0
        cur_msssim = output['msssim_metric'].clone().detach().cpu().numpy() if 'msssim_metric' in output.keys() else 0
        cur_rec = output['rec_loss'].clone().detach().cpu().numpy() if 'rec_loss' in output.keys() else 0
        cur_lpips = output['lpips_loss'].clone().detach().cpu().numpy() if 'lpips_loss' in output.keys() else 0
        ssim_metric.append(cur_ssim)
        msssim_metric.append(cur_msssim)
        rec_loss.append(cur_rec)
        lpips.append(cur_lpips)
    rec_error = np.mean(rec_loss)
    ssim_simi = np.mean(ssim_metric)
    msssim_simi = np.mean(msssim_metric)
    lpips_simi = np.mean(lpips)
    Logger.info(f"rec:{rec_error} ssim:{ssim_simi} msssim:{msssim_simi} lpips:{lpips_simi}")

    Logger.info(f"Validation finished after {time.time() - stime}s.")
    # Logger.info(f"rec_error:{rec_error}, psnr_metric:{psnr_metric}, ssim_error:{ssim_error}")
    Logger.info(f"====================validation  end========================")