import os
import time
import json
import ipdb
import math
import scipy
import io, PIL
import argparse
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from tqdm import tqdm

from src.dataset import get_dataset
from src.model import get_model
from src.utils import get_logger as get_logger
from src.losses.calculate_is_fid import calc_is_fid_from_list

# also disable grad to save memory
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

torch.set_grad_enabled(False)
# DEVICE = torch.device(f"cuda:0"
#                       if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device(f"cuda:{os.environ['CUDA_VISIBLE_DEVICE'].split(',')[0]}"
#                       if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")
print(f"Valid device: {DEVICE}")

get_logger("", "", False)
def get_opt_from_yaml(path):
    assert os.path.exists(path), f"{path} must exists!"
    import yaml
    with open(path, 'r') as f:
        opt = yaml.load(f)
    return opt

def center_crop(img):
    # center crop
    w, h = img.size
    minl = min(h, w)
    left = (w - minl) / 2
    right = (w + minl) / 2
    top = (h - minl) / 2
    bottom = (h + minl) / 2
    img = img.crop((left, top, right, bottom))

    return img

def img2tensor(img, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=Image.BILINEAR),
        transforms.ToTensor()
    ])
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    ret = center_crop(img)
    #     ret.show()
    ret = transform(ret)
    assert len(ret.shape) == 3
    return ret.unsqueeze(0)

def tensor2img(img):
    img = img.clamp(min=0, max=1)
    ret = img.permute(1, 2, 0).detach().cpu().numpy() * 255
    ret = ret.astype(np.uint8)
    ret = Image.fromarray(ret).convert('RGB')
    return ret

def convert_img(img): # img: PIL.Image
    img = np.array(img.convert('RGB'))
    img = scipy.misc.imresize(img, (299, 299), interp='bilinear')
    img = np.cast[np.float32]((-128 + img) / 128.)  # 0~255 -> -1~1
    img = np.expand_dims(img, axis=0).transpose(0, 3, 1, 2)  # NHWC -> NCHW
    return img

def reconstruct_with_dqvae(img, dqvae):
    with torch.no_grad():
        output = dqvae(img, is_training=False, ret_loss=False)
        x_rec = output['x_rec']
    return tensor2img(x_rec[0])

def get_data_loaders(opt):
    data_opt = opt['dataset']
    Logger = get_logger()
    Logger.info(f"Start to get dataset {data_opt['name']}...")
    _, testset = get_dataset(opt)

    assert data_opt['val'] is not None
    Logger.info("testset is Valid.")
    test_sampler = torch.utils.data.SequentialSampler(testset)
    testloader = torch.utils.data.DataLoader(
        testset,
        sampler=test_sampler,
        batch_size=data_opt['batch_size'],
        num_workers=data_opt['num_workers'],
        pin_memory=data_opt['pin_memory'],
        shuffle=data_opt['shuffle'], drop_last=True)

    return testloader, test_sampler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reconstruction_IS/FID")
    parser.add_argument('--opt', default=None, type=str,required=True,
                        help="config file path")
    parser.add_argument('--ckpt', default=None, type=str, required=True,
                        help="checkpoint pth file path")
    parser.add_argument('--if_save', default=False, type=bool,
                        help="If save reconstruct images. Default=False.")
    # Default
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--n_split', default=10)
    args = parser.parse_args()

    s = time.time()
    # Load model
    print(f"Start to load model: {args.ckpt}...")
    opt = get_opt_from_yaml(args.opt)
    opt['model']['checkpoint_path'] = args.ckpt
    model, start_step = get_model(opt)
    dqvae = model.to(DEVICE).eval()

    # Load validate dataset
    # opt['dataset']['name'] += '_EXT'
    testloader, testsampler = get_data_loaders(opt)

    # create save path if need
    if args.if_save:
        save_path_rec = args.ckpt[:-4]+"_recframes"
        save_path_gt = args.ckpt[:-4]+"_gtframes"
        os.makedirs(save_path_rec, exist_ok=True)
        os.makedirs(save_path_gt, exist_ok=True)
        print(f"Start to reconstruct {len(testloader)} frames and save in {save_path_gt}.")
    else:
        save_path = None
        print(f"Start to reconstruct {len(testloader)} frames and do not save.")

    # start to reconstruct
    xs, xrecs = [], []
    tot_iter = len(testloader)
    testloader_iter = enumerate(testloader)
    for step in tqdm(range(tot_iter)):
        _, val_inputs = next(testloader_iter)
        with torch.no_grad():
            val_inputs = [item.to(DEVICE) for item in val_inputs]
            if opt['model']['name'] in ['MoCoVQVAE_wCD', 'MoCoVQVAE_wCD_shareCB', 'MoCoVQVAE_wID',
                                        'MoCoVQVAEwCDsCB_mo', 'MoCoVQVAEwCDsCB_como2']:
                xfull, xbg, xid, xmo = val_inputs
                val_inputs = [xfull, xfull, xfull, xmo]
            output = model(val_inputs, is_training=False, writer=None)

        B, T, _, _, _ = val_inputs[0].shape
        for b in range(B):
            if args.if_save:
                os.makedirs(os.path.join(save_path_gt, "{:06d}_{:03d}".format(step, b)), exist_ok=True)
                os.makedirs(os.path.join(save_path_rec, "{:06d}_{:03d}".format(step, b)), exist_ok=True)
            for t in range(T):
                x = tensor2img(val_inputs[0][b][t])
                xrec = tensor2img(output['x_rec'][b][t])
                if args.if_save:
                    x.save(os.path.join(save_path_gt, "{:06d}_{:03d}".format(step, b), "{:03d}.png".format(t)))
                    xrec.save(os.path.join(save_path_rec, "{:06d}_{:03d}".format(step, b), "{:03d}.png".format(t)))
                xs.append(convert_img(x))
                xrecs.append(convert_img(xrec))
        if (step + 1) * B >= 1000:
            break
    xs = torch.Tensor(np.concatenate(xs[:50000], axis=0))
    xrecs = torch.Tensor(np.concatenate(xrecs[:50000], axis=0))

    print(f"Start to calculate is, fid...")
    print(f"Valid xs:{xs.shape[0]}, xrecs: {xrecs.shape[0]}")
    (IS_mean, IS_std, FID) = calc_is_fid_from_list(xs, xrecs, args.batch_size, args.n_split, DEVICE)
    print(f"FINAL RESULTS:")
    print(f"    IS_mean: {IS_mean}")
    print(f"    IS_std: {IS_std}")
    print(f"    FID: {FID}")
    print(f"    Take time: {time.time()-s}s")