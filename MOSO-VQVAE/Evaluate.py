import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from piqa import PSNR
from piqa import SSIM
from piqa import LPIPS
from einops import rearrange

from ipdb import set_trace as st

psnr = PSNR().cuda()
ssim = SSIM().cuda()
lpips = LPIPS().cuda()

def calculate(path1, path2, size):
    x= np.array(Image.open(path1).resize((size, size)), dtype=np.float32)
    y = np.array(Image.open(path2).resize((size, size)), dtype=np.float32)
    x = torch.tensor(x).cuda()
    y = torch.tensor(y).cuda()

    val_psnr = psnr(x / 255., y / 255.)

    x = rearrange(x, 'h w c -> c h w').unsqueeze(0)
    y = rearrange(y, 'h w c -> c h w').unsqueeze(0)
    val_ssim = ssim(x / 255., y / 255.)
    val_lpips = lpips(x / 255., y / 255.)

    return val_psnr.cpu(), val_ssim.cpu(), val_lpips.cpu()

def get_tar_files(dir):
    v2frames = dict({})
    for video in sorted(os.listdir(dir)):
        frames = os.listdir(os.path.join(dir, video))
        v2frames[video] = [os.path.join(dir, video, f) for f in sorted(frames)]
    return v2frames

import argparse
parser = argparse.ArgumentParser(description="VQVAES")
parser.add_argument('--real', type=str)
parser.add_argument('--fake', type=str)
parser.add_argument('--record_file', type=str, default=None)
args = parser.parse_args()

tar_v2f = get_tar_files(args.real)
src_v2f = get_tar_files(args.fake)

if args.record_file is not None:
    f = open(args.record_file, 'a')

psnrs, ssims, lpipses = [], [], []
td = tqdm(tar_v2f.keys())
for v in td:
    v_psnr, v_ssim, v_lpips = [], [], []
    for j in range(len(tar_v2f[v])):
        p, s, l = calculate(tar_v2f[v][j], src_v2f[v][j], 64)
        v_psnr.append(p.numpy())
        v_ssim.append(s.numpy())
        v_lpips.append(l.detach().numpy())

        if args.record_file is not None:
            tar_path = tar_v2f[v][j].replace(
                '/home/zhongguokexueyuanzidonghuayanjiusuo/mzsun/codes/MoCo/datasets/', '')
            src_path = src_v2f[v][j].replace(
                '/home/zhongguokexueyuanzidonghuayanjiusuo/mzsun/codes/MoCo/datasets/', '')
            f.write("{:03f} {:05f} {:05f} {} {} \n".format(p.numpy(), s.numpy(), l.detach().numpy(), tar_path, src_path))

            if j % 10 == 0:
                f.flush()

    psnrs.append(np.mean(v_psnr))
    ssims.append(np.mean(v_ssim))
    lpipses.append(np.mean(v_lpips))
    td.set_description('{:.2f}-{:.3f}-{:.3f}'.format(np.mean(psnrs), np.mean(ssims), np.mean(lpipses)))

    if len(psnrs) >= 256:
        break

print("PSNR: ", np.mean(psnrs))
print("SSIM: ", np.mean(ssims))
print("LPIPS: ", np.mean(lpipses))



