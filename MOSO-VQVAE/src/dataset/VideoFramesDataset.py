import os
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


class VideoFramesDataset(Dataset):
    def __init__(self, datapath, idspath, img_size, num_frames):
        super().__init__()

        self.img_size = img_size
        self.json_path = idspath
        self.frame_path = datapath
        self.num_frames = num_frames

        self.video_ids = json.load(open(self.json_path, 'r'))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])

        logger = get_logger()
        logger.info(f"{len(self.video_ids)} videos from datapath {datapath},  "
                    f"img_size: {img_size}, num_frames: {num_frames}")


    def __len__(self):
        return len(self.video_ids)

    def skip_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        # random select 4 continuous frames
        imgs = []
        cur_path = os.path.join(self.frame_path, video_id)
        files = sorted(os.listdir(cur_path))
        if len(files) == self.num_frames:
            start = 0
        else:
            start = np.random.choice(range(len(files) - self.num_frames))

        for file in files[start : start + self.num_frames]:
            img_path = os.path.join(cur_path, file)
            img = Image.open(img_path)
            img = random_crop(img)
            cur_img = self.transform(img).unsqueeze(0)
            imgs.append(cur_img)

        # concate
        ret_imgs = torch.cat(imgs, dim=0)
        return ret_imgs # [B,T,C,H,W]


# if __name__ == "__main__":
#     ds = VideoFramesDataset(datapath='/home/zhongguokexueyuanzidonghuayanjiusuo/datasets/msrvtt/frames',
#                              idspath='/home/zhongguokexueyuanzidonghuayanjiusuo/datasets/msrvtt/train_frames_ids.json',
#                              lq_img_size=128, gt_img_size=256)
#     lq_imgs, gt_imgs = ds[0]
#     print(lq_imgs.shape)
#     print(gt_imgs.shape)
