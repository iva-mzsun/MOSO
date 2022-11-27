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

class Generator(object):
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
        self.unknown_number_in_the_beginning = torch.sum(init_tokens == mask_token_id, dim=-1) #[B]

        # Initializes states
        # The position of the decoding loop in the length dimension.
        self.cur_step = start_step
        self.tot_steps = tot_steps
        # The active sequence log probabilities and finished sequence scores.
        self.cur_tokens = init_tokens  # [B, L]
        self.final_tokens = repeat(init_tokens, 'b l -> b n l', n=tot_steps)  # [B, N, L], N is the tot steps

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
        if self.gen_opt.get('topk', -1) == -1:
            sampled_ids = Categorical(logits=logits_flatten).sample().to(torch.int32)
        else:
            print(f"WARNING: sample from topk: {self.gen_opt.topk}")
            bl, v = logits_flatten.shape
            sorted_logits_flatten, _ = torch.sort(logits_flatten, dim=-1)
            cut_off = torch.gather(sorted_logits_flatten, -1,
                                   (torch.ones(bl, 1) * (v - self.gen_opt.topk)).long())
            logits_flatten[logits_flatten < cut_off] = -float("inf")
            sampled_ids = Categorical(logits=logits_flatten).sample().to(torch.int32)
        sampled_ids = rearrange(sampled_ids, '(b l) -> b l', b=logits.shape[0])

        # Maintain PRE-SAMPLED tokens, just updates the masked tokens.
        known_map = (cur_tokens != self.mask_token_id)
        unknown_map = (cur_tokens == self.mask_token_id)
        sampled_ids = sampled_ids * unknown_map + cur_tokens * known_map
        # Defines the mask ratio for the next round. The number to mask out is
        # determined by mask_ratio * unknown_number_in_the_beginning.
        ratio = 1. * (step + 1) / self.tot_steps
        mask_ratio = self.gamma_function(ratio)

        # Updates final seqs with the current sampled_ids.
        self.final_tokens[:, step, :] = sampled_ids
        # Computes the probabilities of each selected tokens.
        probs = torch.softmax(logits, dim=-1)
        selected_probs = torch.gather(probs, -1, sampled_ids.unsqueeze(-1).to(torch.int64)).squeeze(-1)  # [B, L]
        # Ignores the tokens given in the input by overwriting their confidence.
        selected_probs = selected_probs.masked_fill(known_map, float('inf'))
        # Gets mask lens for each sample in the batch according to the mask ratio.
        mask_len = torch.floor(self.unknown_number_in_the_beginning * mask_ratio).unsqueeze(1)  # [B, 1]
        # Keeps at least one of prediction in this round and also masks out at least
        # one and for the next iteration
        mask_len = torch.max(torch.ones_like(mask_len),
                             torch.min(mask_len, torch.sum(unknown_map, dim=-1, keepdim=True) - 1))
        masking = self.mask_by_random_topk2(mask_len, selected_probs, self.gen_opt.temperature)
        # masking = self.mask_by_random_topk(mask_len, selected_probs, self.gen_opt.temperature)
        # Masks tokens with lower confidence.
        sampled_ids = sampled_ids.masked_scatter_(masking,
                                                  (torch.ones_like(masking) * self.mask_token_id).to(torch.int32))

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

    def mask_by_topk(self, mask_len, probs):
        """
        Args:
            mask_len: [B, L], the number to mask for each batch
            probs: [B, L], the probabilities associated with each entry
        Returns:
            A binary masking map [batch_size, seq_len].
        """
        confidence = probs
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.gather(sorted_confidence, -1, mask_len.to(torch.int64))
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking

    def mask_by_random_topk(self, mask_len, probs, temperature=0):
        """
        Args:
            mask_len: [B, L], the number to mask for each batch
            probs: [B, L], the probabilities associated with each entry, [0, 1]
        Returns:
            A binary masking map [batch_size, seq_len].
        """
        # randomness
        randv = torch.rand(probs.shape)
        # add randomness
        confidence = probs + temperature * randv
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.gather(sorted_confidence, -1, mask_len.to(torch.int64))
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking

    def mask_by_random_topk2(self, mask_len, probs, temperature=0):
        """
        Args:
            mask_len: [B, L], the number to mask for each batch
            probs: [B, L], the probabilities associated with each entry, [0, 1]
        Returns:
            A binary masking map [batch_size, seq_len].
        """
        # randomness
        gumbelv = torch.from_numpy(np.random.gumbel(size=probs.shape))
        # add randomness
        confidence = torch.log(probs) + temperature * gumbelv
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.gather(sorted_confidence, -1, mask_len.to(torch.int64))
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking


class Generator_mo(object):
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
            cur_generator = Generator(gen_opt, cur_init_tokens, start_step, tot_steps, mask_token_id, gamma_function)
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
