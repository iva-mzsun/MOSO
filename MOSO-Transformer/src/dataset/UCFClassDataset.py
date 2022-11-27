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


class UCFClassDataset(Dataset):
    def __init__(self, tokens_dir, class2id):
        super().__init__()

        self.tokens_dir = tokens_dir
        self.videos = os.listdir(tokens_dir)
        self.class2id = json.load(open(class2id, 'r'))
        self.items = []
        for cvideo in self.videos:
            for item in os.listdir(os.path.join(tokens_dir, cvideo)):
                self.items.append((cvideo, item))

    def __len__(self):
        return len(self.items)

    def skip_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def __getitem__(self, index):
        item = self.items[index]

        cur_class = self.class2id[item[0].split('_')[1]]
        ctoken = np.load(os.path.join(self.tokens_dir, item[0], item[1]),
                         allow_pickle=True)
        bg_tokens = ctoken.item()['bg_tokens'] # H, W
        id_tokens = ctoken.item()['id_tokens'] # H, W
        mo_tokens = ctoken.item()['mo_tokens'] # T, H, W

        cur_class = np.array(cur_class).astype(np.int32)
        bg_tokens = bg_tokens.flatten().astype(np.int32)
        id_tokens = id_tokens.flatten().astype(np.int32)
        mo_tokens = mo_tokens.flatten().astype(np.int32)

        return {
            'class': cur_class,
            'bg_tokens': bg_tokens,
            'id_tokens': id_tokens,
            'mo_tokens': mo_tokens
        }