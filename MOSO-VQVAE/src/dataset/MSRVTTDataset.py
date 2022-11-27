import os
import PIL
import json
import numpy as np
from PIL import Image
from random import shuffle

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset


class MsrvttFramesDataset(Dataset):
    def __init__(self, datapath, idspath, img_size):
        super().__init__()
        self.json_path = idspath
        self.frame_path = datapath

        self.video_ids = json.load(open(self.json_path, 'r'))
        # shuffle(self.video_ids)

        # get video_captions
        self.json_dict = json.load(open(self.json_path, 'r'))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor()
        ])


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
        if len(files) == 4:
            start = 0
        else:
            start = np.random.choice(range(len(files) - 4))

        for file in files[start : start + 4]:
            img_path = os.path.join(cur_path, file)
            img = torchvision.datasets.folder.pil_loader(img_path)

            w, h = img.size
            minl = min(h, w)
            left = (w - minl) / 2
            right = (w + minl) / 2
            top = (h - minl) / 2
            bottom = (h + minl) / 2
            img = img.crop((left, top, right, bottom))

            img = self.transform(img).unsqueeze(-3)

            imgs.append(img)

        # concate
        imgs = torch.cat(imgs, dim=-3)

        return imgs


if __name__ == "__main__":
    ds = MsrvttFramesDataset(datapath='/home/zhongguokexueyuanzidonghuayanjiusuo/datasets/msrvtt/frames',
                             idspath='/home/zhongguokexueyuanzidonghuayanjiusuo/datasets/msrvtt/train_frames_ids.json',
                             lq_img_size=128, gt_img_size=256)
    lq_imgs, gt_imgs = ds[0]
    print(lq_imgs.shape)
    print(gt_imgs.shape)
