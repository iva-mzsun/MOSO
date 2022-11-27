import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from piqa import LPIPS
from einops import rearrange

from ipdb import set_trace as st

lpips = LPIPS()

def calculate(path1, path2, size):
    x = np.expand_dims(np.array(Image.open(path1).convert('RGB').resize((size, size)), dtype=np.float32), axis=0) / 255.
    y = np.expand_dims(np.array(Image.open(path2).convert('RGB').resize((size, size)), dtype=np.float32), axis=0) / 255.
    x = rearrange(x, 'b h w c -> b c h w')
    y = rearrange(y, 'b h w c -> b c h w')
    val_lpips = lpips(torch.tensor(x), torch.tensor(y))

    val_psnr = np.mean([
        psnr_metric(x[0][0], y[0][0], data_range=1.0),
        psnr_metric(x[0][1], y[0][1], data_range=1.0),
        psnr_metric(x[0][2], y[0][2], data_range=1.0)
    ])
    val_ssim = np.mean([
        ssim_metric(x[0][0], y[0][0], data_range=1.0),
        ssim_metric(x[0][1], y[0][1], data_range=1.0),
        ssim_metric(x[0][2], y[0][2], data_range=1.0)
    ])

    return val_psnr, val_ssim, val_lpips

def get_tar_files(dir):
    imgs = []
    for video in sorted(os.listdir(dir)):
        frames = os.listdir(os.path.join(dir, video))
        if len(frames) != 40:
            print(dir, video)
            continue
        imgs += [os.path.join(dir, video, f) for f in sorted(frames)]
    return imgs

import argparse
parser = argparse.ArgumentParser(description="VQVAES")
parser.add_argument('--real', type=str)
parser.add_argument('--fake', type=str)
parser.add_argument('--record_file', type=str, default=None)
args = parser.parse_args()

tar_frames = get_tar_files(args.real)
src_frames = get_tar_files(args.fake)

if args.record_file is not None:
    f = open(args.record_file, 'a')

psnrs, ssims, lpipses = [], [], []
print(len(tar_frames), len(src_frames))
assert len(tar_frames) == len(src_frames)
td = tqdm(range(len(tar_frames)))
for i in td:
    p, s, l = calculate(tar_frames[i], src_frames[i], 64)
    psnrs.append(p)
    ssims.append(s)
    lpipses.append(l.detach().numpy())

    if args.record_file is not None:
        tar_path = tar_frames[i].replace(
            '/home/zhongguokexueyuanzidonghuayanjiusuo/mzsun/codes/MoCo/datasets/', '')
        src_path = src_frames[i].replace(
            '/home/zhongguokexueyuanzidonghuayanjiusuo/mzsun/codes/MoCo/datasets/', '')
        f.write("{:03f} {:05f} {:05f} {} {} \n".format(p, s, l.detach().numpy(), tar_path, src_path))

        if i % 10 == 0:
            f.flush()

    td.set_description('{:.2f}-{:.3f}-{:.3f}'.format(np.mean(psnrs), np.mean(ssims), np.mean(lpipses)))

print("PSNR: ", np.mean(psnrs))
print("SSIM: ", np.mean(ssims))
print("LPIPS: ", np.mean(lpipses))

# if args.record_path is not None:
#     json.dump(records, open(args.record_path, 'w'))


