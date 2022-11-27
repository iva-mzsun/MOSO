import os
import ipdb
import PIL
import json
import numpy as np
from PIL import Image
from random import shuffle
from src.utils import  get_logger

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

def random_crop(cur_img):
    width, height = cur_img.size
    if width % 2 == 1:
        width -= 1
    if height % 2 == 1:
        height -= 1
    # random crop
    if width == height:
        cur_img = np.array(cur_img).astype(np.float32)
        return np.expand_dims(cur_img, axis=0)
    elif width < height:
        diff = height - width
        move = np.random.choice(diff) - diff // 2
        left, right = 0, width
        top = (height - width) // 2 + move
        bottom = (height + width) // 2 + move
    else:
        diff = width - height
        move = np.random.choice(diff) - diff // 2
        top, bottom = 0, height
        left = (width - height) // 2 + move
        right = (width + height) // 2 + move

    cur_img = cur_img.crop((left, top, right, bottom))
    return cur_img

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

def get_img_from_path(img_path, transform=None):
    img = Image.open(img_path)
    img = center_crop(img)
    if transform is None:
        return img
    else:
        return transform(img)

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

class VideoDiffFramesDataset_EXT_FirstTwoFrames(Dataset):
    def __init__(self, datapath, idspath, used_fps, img_size, num_frames, limit, ret_mode='dict'):
        super().__init__()
        self.limit = limit
        self.boarden = 0.4
        self.lower_bound = max(0, self.limit - self.boarden)
        self.upper_bound = min(1, self.limit + self.boarden)
        self.ret_mode = ret_mode

        self.img_size = img_size
        self.frame_path = datapath
        self.num_frames = num_frames
        logger = get_logger()

        if idspath is not None:
            self.video_ids = json.load(open(idspath, 'r'))
        else:
            assert used_fps
            used_fps = used_fps.split(',')
            logger.info(f"*** Used FPS {used_fps} for datapath: {datapath}")
            self.video_ids = []
            for cfps in used_fps:
                cfps_path = os.path.join(datapath, cfps)
                cvideos = [os.path.join(cfps, v) for v in os.listdir(cfps_path)]
                self.video_ids += cvideos
        shuffle(self.video_ids)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor()
        ])

        # get all video samples to easily extract
        self.items = []
        for video_id in self.video_ids:
            cur_path = os.path.join(self.frame_path, video_id)
            filen = len(os.listdir(cur_path))
            if filen < self.num_frames + 2:
                continue
            else:
                for start in range(0, filen-1-self.num_frames, self.num_frames):
                    self.items.append([video_id, start])
                if filen % self.num_frames != 0: # To utilize all frames, count last clip from the end frame to previous ones.
                    self.items.append([video_id, filen-2-self.num_frames])

        logger.info(f"*** {len(self.video_ids)} videos and {len(self.items)} items from datapath {datapath},  "
                    f"img_size: {img_size}, num_frames: {num_frames}")

    def __len__(self):
        return len(self.items)

    def skip_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def __getitem__(self, index):
        video_id, start = self.items[index]

        # random select 4 continuous frames
        cur_path = os.path.join(self.frame_path, video_id)
        files = sorted(os.listdir(cur_path))

        # start to load img
        pre_img = get_img_from_path(os.path.join(cur_path, files[start + 1]),
                                    transform=self.transform)
        nxt_img = get_img_from_path(os.path.join(cur_path, files[start + 1]),
                                    transform=self.transform)
        imgs, imgs_bg, imgs_id, imgs_mo = [], [], [], []
        for file in files[start + 2: start + 2 + self.num_frames]:
            cur_img = nxt_img
            if file == files[start + 2]:
                nxt_img = get_img_from_path(os.path.join(cur_path, file),
                                            transform=self.transform)

            cur_diff = cur_img * 2 - pre_img - nxt_img
            max_diff = torch.max(torch.abs(cur_diff), dim=0)[0]
            id_mask = (max_diff >= self.lower_bound) * (max_diff <= self.upper_bound)
            img_id = id_mask[None, :, :] * cur_img
            img_bg = cur_img - img_id

            imgs.append(cur_img.unsqueeze(0))
            imgs_bg.append(img_bg.unsqueeze(0))
            imgs_id.append(img_id.unsqueeze(0))
            imgs_mo.append(cur_diff.unsqueeze(0))

        # concate, [T,C,H,W]
        ret_img = torch.cat(imgs, dim=0)
        ret_img_bg = torch.cat(imgs_bg, dim=0)
        ret_img_id = torch.cat(imgs_id, dim=0)
        ret_img_mo = torch.cat(imgs_mo, dim=0)

        if self.ret_mode.lower() == 'list':
            return [ret_img, ret_img_bg, ret_img_id, ret_img_mo]
        elif self.ret_mode.lower() == 'dict':
            return {
                'ret_img': ret_img,
                'ret_img_bg': ret_img_bg,
                'ret_img_id': ret_img_id,
                'ret_img_mo': ret_img_mo,
                'video_id': video_id,
                'start': start
            }
        else:
            raise NotImplementedError
