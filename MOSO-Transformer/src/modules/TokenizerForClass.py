import math
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as st
from einops import rearrange, repeat

from src.utils import *
from moco_vqvae.src.model import get_model
from moco_vqvae.src.utils import get_logger

class Class_Tokenizer(nn.Module):
    def __init__(self, vqvae_opt, train_opt, tokenizer_opt, load_vqvae):
        super().__init__()
        self.train_opt = train_opt

        # if warp video tokens
        self.if_warp_token = tokenizer_opt.if_warp_token

        # Load VQVAE model
        if load_vqvae:
            vqvae_cfg = get_dict_from_yaml(vqvae_opt.config_file)
            vqvae_cfg['model']['checkpoint_path'] = vqvae_opt.checkpoint
            vqvae, _ = get_model(vqvae_cfg)
            self.vqvae = vqvae.eval().to(train_opt.device)
            self.bg_vocabs = vqvae._vq_ema_bg.num_embeddings
            self.id_vocabs = vqvae._vq_ema_id.num_embeddings
            self.mo_vocabs = vqvae._vq_ema_mo.num_embeddings
        else:
            print(f"WARNING: !!NOT LOAD VQVAE")
            vqvae_cfg = get_dict_from_yaml(vqvae_opt.config_file)
            self.vqvae = None
            self.bg_vocabs = vqvae_cfg['model']['num_embeddings_c']
            self.id_vocabs = vqvae_cfg['model']['num_embeddings_c']
            self.mo_vocabs = vqvae_cfg['model']['num_embeddings_m']

        # class vocabs
        self.id2class = json.load(open(tokenizer_opt.id2class, 'r'))
        self.class_vocabs = len(self.id2class.items())

        # number of frames
        self.num_frames = tokenizer_opt.num_frames

        # All vocabs
        pre_vocabs = self.bg_vocabs + self.id_vocabs + \
                     self.mo_vocabs + self.class_vocabs
        self.command_tokens = {
            'MASK': pre_vocabs + 0,
            'SOBG': pre_vocabs + 1,
            'EOBG': pre_vocabs + 2,
            'SOID': pre_vocabs + 3,
            'EOID': pre_vocabs + 4,
            'SOMO01': pre_vocabs + 5,
            'EOMO01': pre_vocabs + 6,
            'SOMO02': pre_vocabs + 7,
            'EOMO02': pre_vocabs + 8,
            'SOMO03': pre_vocabs + 9,
            'EOMO03': pre_vocabs + 10,
            'SOMO04': pre_vocabs + 11,
            'EOMO04': pre_vocabs + 12,
            'SOMO05': pre_vocabs + 13,
            'EOMO05': pre_vocabs + 14,
            'SOMO06': pre_vocabs + 15,
            'EOMO06': pre_vocabs + 16,
            'SOMO07': pre_vocabs + 17,
            'EOMO07': pre_vocabs + 18,
            'SOMO08': pre_vocabs + 19,
            'EOMO08': pre_vocabs + 20,
            'SOMO09': pre_vocabs + 21,
            'EOMO09': pre_vocabs + 22,
            'SOMO10': pre_vocabs + 23,
            'EOMO10': pre_vocabs + 24,
            'SOMO11': pre_vocabs + 25,
            'EOMO11': pre_vocabs + 26,
            'SOMO12': pre_vocabs + 27,
            'EOMO12': pre_vocabs + 28,
            'SOMO13': pre_vocabs + 29,
            'EOMO13': pre_vocabs + 30,
            'SOMO14': pre_vocabs + 31,
            'EOMO14': pre_vocabs + 32,
            'SOMO15': pre_vocabs + 33,
            'EOMO15': pre_vocabs + 34,
            'SOMO16': pre_vocabs + 35,
            'EOMO16': pre_vocabs + 36,
        }
        self.command_vocabs = len(self.command_tokens)
        self.vocab_size = self.class_vocabs + \
                          self.bg_vocabs + self.id_vocabs + \
                          self.mo_vocabs + self.command_vocabs

        # general vocabs
        # intorporated codebook: [bg_vocab, id_vocab, mo_vocab, class_vocab, cmd_vovab]
        self.bg_vocab_s = 0
        self.id_vocab_s = self.bg_vocabs
        self.mo_vocab_s = self.bg_vocabs + self.id_vocabs
        self.class_vocab_s = self.bg_vocabs + self.id_vocabs + self.mo_vocabs
        self.cmd_vocab_s = self.class_vocabs + self.bg_vocabs + self.id_vocabs + self.mo_vocabs

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
        newbg = xbg.clone()
        newid = xid.clone()
        newmo = xmo.clone()
        newbg[newbg < self.class_vocab_s] += self.bg_vocab_s
        newid[newid < self.class_vocab_s] += self.id_vocab_s
        newmo[newmo < self.class_vocab_s] += self.mo_vocab_s

        return newbg, newid, newmo

    def rewarp(self, item):
        if len(item.shape) == 1: # [L]
            return item[1:-1]
        elif len(item.shape) == 2: # [B, L]
            return item[:, 1:-1]
        elif len(item.shape) == 3: # [B, L, V]
            return item[:, 1:-1, :]
        else:
            raise NotImplementedError

    def warp_tokens(self, x_bg, x_id, x_mo):
        '''
            x_bg: [B, seq_bg_L]
            x_id: [B, seq_id_L]
            x_mo: [B, T * L]
        '''
        if self.if_warp_token is False:
            return x_bg, x_id, x_mo

        B = x_mo.shape[0]
        new_bg = torch.cat([
            torch.ones(B, 1).to(self.train_opt.device) * self.command_tokens['SOBG'],
            x_bg.to(self.train_opt.device),
            torch.ones(B, 1).to(self.train_opt.device) * self.command_tokens['EOBG']
        ], dim=1).to(torch.long)

        new_id = torch.cat([
            torch.ones(B, 1).to(self.train_opt.device) * self.command_tokens['SOID'],
            x_id.to(self.train_opt.device),
            torch.ones(B, 1).to(self.train_opt.device) * self.command_tokens['EOID']
        ], dim=1).to(torch.long)

        new_mo = torch.cat([
            torch.ones(B, 1).to(self.train_opt.device) * self.command_tokens['SOMO0'],
            x_mo.to(self.train_opt.device),
            torch.ones(B, 1).to(self.train_opt.device) * self.command_tokens['EOMO0']
        ], dim=1).to(torch.long)

        return new_bg, new_id, new_mo

    def warp_masks(self, maskbg, maskid, maskmo):
        '''
            mask_bg: [B, seq_bg_L]
            mask_id: [B, seq_id_L]
            mask_mo: [B, seq_mo_L]
        '''
        if len(maskbg.shape) == 1:
            new_maskbg = torch.cat([torch.zeros(1), torch.tensor(maskbg), torch.zeros(1)], dim=0).to(torch.int32)
            new_maskid = torch.cat([torch.zeros(1), torch.tensor(maskid), torch.zeros(1)], dim=0).to(torch.int32)
            new_maskmo = torch.cat([torch.zeros(1), torch.tensor(maskmo), torch.zeros(1)], dim=0).to(torch.int32)
        elif len(maskbg.shape) == 2:
            B, _ = maskbg.shape
            new_maskbg = torch.cat([torch.zeros(B, 1), torch.tensor(maskbg), torch.zeros(B, 1)], dim=1).to(torch.int32)
            new_maskid = torch.cat([torch.zeros(B, 1), torch.tensor(maskid), torch.zeros(B, 1)], dim=1).to(torch.int32)
            new_maskmo = torch.cat([torch.zeros(B, 1), torch.tensor(maskmo), torch.zeros(B, 1)], dim=1).to(torch.int32)
        else:
            raise NotImplementedError
        new_maskbg = new_maskbg.to(self.train_opt.device)
        new_maskid = new_maskid.to(self.train_opt.device)
        new_maskmo = new_maskmo.to(self.train_opt.device)

        return new_maskbg, new_maskid, new_maskmo



