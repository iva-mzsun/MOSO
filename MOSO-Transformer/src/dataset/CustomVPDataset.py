import os
import ipdb
import PIL
import json
import numpy as np
from einops import repeat, rearrange

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset


class CustomVPDataset(Dataset):
    def __init__(self, tokens_16f_dir, tokens_1th_dir, items=None):
        super().__init__()

        self.tokens_16f_dir = tokens_16f_dir
        self.tokens_1th_dir = tokens_1th_dir
        self.videos = os.listdir(tokens_16f_dir)
        if items is None:
            self.items = []
            for cvideo in self.videos:
                for item in os.listdir(os.path.join(tokens_16f_dir, cvideo)):
                    if os.path.exists(os.path.join(tokens_1th_dir, cvideo, item)):
                        self.items.append((cvideo, item))
        else:
            self.items = items

    def __len__(self):
        return len(self.items)

    def skip_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def load_tokens(self, token_path):
        ctoken = np.load(token_path, allow_pickle=True)
        bg_tokens = ctoken.item()['bg_tokens']  # H, W
        id_tokens = ctoken.item()['id_tokens']  # H, W
        mo_tokens = ctoken.item()['mo_tokens']  # T, H, W
        bg_tokens = bg_tokens.flatten().astype(np.int32)
        id_tokens = id_tokens.flatten().astype(np.int32)
        mo_tokens = mo_tokens.flatten().astype(np.int32)
        return bg_tokens, id_tokens, mo_tokens

    def __getitem__(self, index):
        item = self.items[index]

        try:
            bg1, id1, mo1 = self.load_tokens(os.path.join(self.tokens_1th_dir, item[0], item[1]))
            bg16, id16, mo16 = self.load_tokens(os.path.join(self.tokens_16f_dir, item[0], item[1]))
        except:
            return self.skip_sample(index)

        return {
            'bg1': bg1,
            'id1': id1,
            'mo1': mo1,
            'bg16': bg16,
            'id16': id16,
            'mo16': mo16,
            'video_item': item
        }