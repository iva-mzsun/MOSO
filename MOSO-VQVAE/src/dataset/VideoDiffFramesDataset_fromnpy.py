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

class VideoDiffFramesDataset_fromnpy(Dataset):
    def __init__(self, datapath):
        super().__init__()
        logger = get_logger()

        items = os.listdir(datapath)
        shuffle(items)
        self.items = [os.path.join(datapath, item) for item in items]

        logger.info(f"*** {len(self.items)} items from datapath {datapath}")


    def __len__(self):
        return len(self.items)

    def skip_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def __getitem__(self, index):
        item = self.items[index]
        try:
            values = np.load(item, allow_pickle=True).item()

            ret_img = values['ret_img'][0]
            ret_img_bg = values['ret_img_bg'][0]
            ret_img_id = values['ret_img_id'][0]
            ret_img_mo = values['ret_img_mo'][0]
        except:
            return self.skip_sample(index)

        return [ret_img, ret_img_bg, ret_img_id, ret_img_mo]
