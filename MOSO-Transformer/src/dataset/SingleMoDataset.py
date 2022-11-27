import os
import ipdb
import PIL
import json
import numpy as np
from einops import repeat

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset


class SingleMoDataset(Dataset):
    def __init__(self, tokens_dir):
        super().__init__()

        self.tokens_dir = tokens_dir
        self.videos = os.listdir(tokens_dir)
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

        item_path = os.path.join(self.tokens_dir, item[0], item[1])
        ctoken = np.load(item_path, allow_pickle=True)

        bg_tokens = ctoken.item()['bg_tokens'] # H, W
        id_tokens = ctoken.item()['id_tokens'] # H, W
        mo_tokens = ctoken.item()['mo_tokens'] # T, H, W

        bg_tokens = bg_tokens.flatten().astype(np.int32)
        id_tokens = id_tokens.flatten().astype(np.int32)
        mo_tokens = mo_tokens[0].flatten().astype(np.int32)

        return {
            'bg_tokens': bg_tokens,
            'id_tokens': id_tokens,
            'mo_tokens': mo_tokens
        }