import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as st
from einops import rearrange, repeat

from src.utils import *
from MoCoVQVAE.src.model import get_model
from MoCoVQVAE.src.utils import get_logger

class Tokenizer(nn.Module):
    def __init__(self, vqvae_opt, train_opt, tokenizer_opt, load_vqvae):
        super().__init__()
        self.train_opt = train_opt

        # Load VQVAE model
        if load_vqvae:
            vqvae_cfg = get_dict_from_yaml(vqvae_opt.config_file)
            vqvae_cfg['model']['checkpoint_path'] = vqvae_opt.checkpoint
            vqvae, _ = get_model(vqvae_cfg)
            self.vqvae = vqvae.eval().to(train_opt.device)
            self.video_vocabs = vqvae._vq_ema.num_embeddings
        else:
            print(f"WARNING: !!NOT LOAD VQVAE")
            vqvae_cfg = get_dict_from_yaml(vqvae_opt.config_file)
            self.vqvae = None
            self.video_vocabs = vqvae_cfg['model']['num_embeddings']

        # number of frames
        self.num_frames = tokenizer_opt.num_frames

        # All vocabs
        pre_vocabs = self.video_vocabs
        self.command_tokens = {
            'MASK': pre_vocabs + 0,
            'FPS2': pre_vocabs + 1,
            'FPS4': pre_vocabs + 2,
            'FPS8': pre_vocabs + 3,
            'FPS16': pre_vocabs + 4,
            'FPS32': pre_vocabs + 5,
        }
        pre_cmd_vocabs = len(self.command_tokens)
        for i in range(self.num_frames):
            self.command_tokens[f'SMO{i+1}'] = pre_vocabs + pre_cmd_vocabs + i
        self.command_vocabs = len(self.command_tokens)
        self.vocab_size = self.video_vocabs + self.command_vocabs

        # intorporated codebook: [video_vocab, cmd_vovab]
        self.video_vocab_s = 0
        self.cmd_vocab_s = self.video_vocabs

    def get_hmo(self, batchsize):
        hmo = []
        for i in range(self.num_frames):
            hmo.append(self.command_tokens[f'SMO{i+1}'])
        hmo = torch.tensor(hmo, dtype=torch.int32)
        hmo = repeat(hmo, 'l -> b l', b = batchsize)
        return hmo

    def decode(self, xbg, xid, xmo):
        """ Decode generated tokens
                xbg: [B, L]
                xid: [B, L]
                xmo: [B, T * L]
        """

        bgl, idl, mol = xbg.shape[1], xid.shape[1], xmo.shape[1]
        mol = mol // self.num_frames

        xbg = rearrange(xbg, "b (h w) -> b h w", h=int(np.sqrt(bgl)))
        xid = rearrange(xid, "b (h w) -> b h w", h=int(np.sqrt(idl)))
        xbg = repeat(xbg, "b h w -> b t h w", t=1)
        xid = repeat(xid, "b h w -> b t h w", t=1)
        xmo = rearrange(xmo, "b (t h w) -> b t h w",
                        t=self.num_frames, h=int(np.sqrt(mol)))
        cur_x = self.vqvae._decode(xbg, xid, xmo)
        return cur_x

    def random_mask(self, x_tokens, rate):
        ''' random mask L*rate tokens for each batch of x_tokens
            xmask: 1 if masked
            x_tokens: [B, L]
            rate: a scalar 'r'
        '''
        B, L = x_tokens.shape
        tot_mask_num = min(L - 1, max(1, int(L * rate)))

        mask_ids = torch.multinomial(torch.ones(B, L), tot_mask_num, replacement=False) # [B, tot_mask_num]
        xmask = torch.sum(F.one_hot(mask_ids, L), dim=1).to(x_tokens.device) # [B, L]

        # replace selected masked tokens with token 'MASK'
        x = x_tokens * (1 - xmask) + xmask * self.command_tokens['MASK']

        return x, xmask

    def random_mask_mo(self, mo_tokens, rate):
        """Separately mask L*rate tokens for T groups of mo_tokens
            mo_tokens: [B, T * L]
        """
        T = self.num_frames
        mo_l = mo_tokens.shape[1] // T
        masked_mo, momasks = [], []
        for i in range(T):
            cur_tokens = mo_tokens[:, i*mo_l:(i+1)*mo_l]
            x, xmask = self.random_mask(cur_tokens, rate)
            masked_mo.append(x)
            momasks.append(xmask)
        masked_mo = torch.cat(masked_mo, dim=1)
        momasks = torch.cat(momasks, dim=1)
        return masked_mo, momasks

    def pad_vocab(self, xbg, xid, xmo):
        newbg, newid, newmo = None, None, None
        if xbg is not None:
            newbg = xbg.clone()
        if xid is not None:
            newid = xid.clone()
        if xmo is not None:
            newmo = xmo.clone()

        return newbg, newid, newmo

