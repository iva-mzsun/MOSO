import math
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributions import Categorical

from einops import rearrange, repeat
from src.utils import *

from ipdb import set_trace as st

class Refiner_woCONFI(object):
    def __init__(self, gen_opt, init_tokens, start_step, tot_steps, mask_token_id, gamma_function):
        """Fast decoding for iterative generation.

            Args:
              inputs: int32 array: [batch_size, seq_length] input sequence of masked
                tokens, where the masking tokens is defined by mask_token_id.
              rng: jnp.DeviceArray: sampling random state.
              tokens_to_logits: decoder function taking single token slices and cache and
                returning logits and updated cache.
              mask_token_id: int: [Mask] token id.
              num_iter: int: default is 12.
              start_iter: int: default is 0.
              choice_temperature: float: temperature to control the randomness of masking.
              mask_scheduling_method: masking method string. See mask_schedule.py for
                details.

            Returns:
               [batch_size, num_iter, seq_length] output sequence of tokens in all
                 iterations.
            """

        self.gen_opt = gen_opt
        self.mask_token_id = mask_token_id
        self.gamma_function = gamma_function

        # Init mask in each step
        self.masks = self.get_masks_for_each_step(init_tokens, tot_steps)  # [B, N, L], set 1 if masked

        # Initializes states
        # The position of the decoding loop in the length dimension.
        self.cur_step = start_step
        self.tot_steps = tot_steps
        # The active sequence log probabilities and finished sequence scores.
        cur_mask = self.masks[:, start_step, :]  # [B, L]
        # Masks tokens with lower confidence.
        cur_tokens = init_tokens.masked_scatter_(cur_mask,
                                                (torch.ones_like(cur_mask) * mask_token_id).to(torch.int32))
        self.cur_tokens = cur_tokens  # [B, L]
        self.final_tokens = repeat(init_tokens, 'b l -> b n l', n=tot_steps)  # [B, N, L], N is the tot steps

    def get_masks_for_each_step(self, init_tokens, tot_steps):
        B, L = init_tokens.shape
        left = torch.ones_like(init_tokens, dtype=torch.float32) # All tokens are masked
        MASK_PER_STEP = L // tot_steps
        masks = []
        for i in range(tot_steps - 1):
            select = torch.multinomial(left, MASK_PER_STEP,
                                       replacement=False) # [B, MASK_PER_STEP]
            uncover = torch.sum(F.one_hot(select, L), dim=1) # [B, L]
            left = left - uncover # unmask select tokens
            masks.append(uncover.unsqueeze(1)) # NOTE: Masks in Refiner is independent
        masks.append(left.unsqueeze(1))
        masks = torch.cat(masks, dim=1).to(torch.int32) # [B, N, L]
        return masks

    def generate_one_step(self, logits):
        """
            logits: [B, L, V], V is the vocab_size and is in consistent with init_tokens
            logits is the Transformer output after feeding cur_tokens
        """
        # Current tokens: [batch_size, seq_length]
        step = self.cur_step
        cur_tokens = self.cur_tokens

        # Samples the ids using categorical sampling: [batch_size, seq_length].
        logits_flatten = rearrange(logits, 'b l v -> (b l) v')
        sampled_ids = Categorical(logits=logits_flatten).sample().to(torch.int32)
        sampled_ids = rearrange(sampled_ids, '(b l) -> b l', b=logits.shape[0])

        # Maintain PRE-SAMPLED tokens, just updates the masked tokens.
        known_map = (cur_tokens != self.mask_token_id)
        unknown_map = (cur_tokens == self.mask_token_id)
        # sampled_ids = sampled_ids.masked_scatter_(known_map, cur_tokens)
        sampled_ids = sampled_ids * unknown_map + cur_tokens * known_map
        # Updates final seqs with the current sampled_ids.
        self.final_tokens[:, step, :] = sampled_ids

        cur_mask = self.masks[:, step, :] # [B, L]
        # Masks tokens with lower confidence.
        sampled_ids = sampled_ids.masked_scatter_(cur_mask,
                                                  (torch.ones_like(cur_mask) * self.mask_token_id).to(torch.int32))

        # Updatates state
        self.cur_tokens = sampled_ids
        self.cur_step += 1

    def get_cur_tokens(self):
        return self.cur_tokens

    def get_final_tokens(self):
        return self.final_tokens

    def unfinished(self):
        # return 0 as finished, 1 as unfinished
        return self.cur_step < self.tot_steps


class Refiner_mo_woCONFI(object):
    def __init__(self, num_frames, gen_opt, init_tokens, start_step, tot_steps, mask_token_id, gamma_function):
        """Fast decoding for iterative generation.

            Args:
              inputs: int32 array: [batch_size, seq_length] input sequence of masked
                tokens, where the masking tokens is defined by mask_token_id.
              rng: jnp.DeviceArray: sampling random state.
              tokens_to_logits: decoder function taking single token slices and cache and
                returning logits and updated cache.
              mask_token_id: int: [Mask] token id.
              num_iter: int: default is 12.
              start_iter: int: default is 0.
              choice_temperature: float: temperature to control the randomness of masking.
              mask_scheduling_method: masking method string. See mask_schedule.py for
                details.

            Returns:
               [batch_size, num_iter, seq_length] output sequence of tokens in all
                 iterations.
            """

        self.num_frames = num_frames
        self.mo_generators = []

        T = self.num_frames
        L = init_tokens.shape[1] // T
        for i in range(T):
            cur_init_tokens = init_tokens[:, i*L:(i+1)*L]
            cur_generator = Refiner_woCONFI(gen_opt, cur_init_tokens, start_step, tot_steps, mask_token_id, gamma_function)
            self.mo_generators.append(cur_generator)

    def generate_one_step(self, logits):
        """
            logits: [B, T * L, V], V is the vocab_size and is in consistent with init_tokens
            logits is the Transformer output after feeding cur_tokens
        """
        T = self.num_frames
        L = logits.shape[1] // T
        assert logits.shape[1] == T * L

        for i in range(T):
            cur_logit = logits[:, i*L:(i+1)*L, :]
            self.mo_generators[i].generate_one_step(cur_logit)

    def get_cur_tokens(self):
        # cur_tokens: B, T * L
        cur_tokens = torch.cat([self.mo_generators[i].cur_tokens
                                for i in range(self.num_frames)], dim=1)
        return cur_tokens

    def get_final_tokens(self):
        # final_tokens: B, STEPS, T * L
        final_tokens = torch.cat([self.mo_generators[i].final_tokens
                                  for i in range(self.num_frames)], dim=2)
        return final_tokens

    def unfinished(self):
        # return 0 as finished, 1 as unfinished
        return self.mo_generators[0].unfinished()
