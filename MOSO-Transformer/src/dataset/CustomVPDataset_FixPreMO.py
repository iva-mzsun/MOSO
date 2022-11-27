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


class CustomVPDataset_FixPreMo(Dataset):
    def __init__(self, tokens_tar_dir, tokens_pre_dir, pre_frame_num, items=None):
        super().__init__()

        self.pre_frame_num = pre_frame_num
        self.tokens_tar_dir = tokens_tar_dir
        self.tokens_pre_dir = tokens_pre_dir
        self.videos = os.listdir(tokens_tar_dir)
        if items is None:
            self.items = []
            for cvideo in self.videos:
                for item in os.listdir(os.path.join(tokens_tar_dir, cvideo)):
                    if os.path.exists(os.path.join(tokens_pre_dir, cvideo, item)):
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
        fix_mo_tokens = mo_tokens[:self.pre_frame_num].flatten().astype(np.int32)
        suf_mo_tokens = mo_tokens[self.pre_frame_num:].flatten().astype(np.int32)
        return bg_tokens, id_tokens, fix_mo_tokens, suf_mo_tokens

    def __getitem__(self, index):
        item = self.items[index]

        try:
            bg_tar, id_tar, base_mo_tar, mo_tar = self.load_tokens(os.path.join(self.tokens_tar_dir, item[0], item[1]))
            bg_pre, id_pre, base_mo_pre, mo_pre = self.load_tokens(os.path.join(self.tokens_pre_dir, item[0], item[1]))
            assert torch.all(torch.tensor(base_mo_pre == base_mo_tar))
        except Exception as e:
            print(e)
            print("Skip sample: ", item[0], item[1])
            return self.skip_sample(index)

        return {
            'bg_tar': bg_tar,
            'id_tar': id_tar,
            'mo_tar': mo_tar,
            'bg_pre': bg_pre,
            'id_pre': id_pre,
            'mo_pre': mo_pre,
            'mo_base': base_mo_tar,
            'video_item': item
        }