import wandb
import deepspeed

import torch.nn.functional as F
from einops import rearrange, repeat
from torch.distributions import Categorical

from src.utils import *
from src.dataset import get_dataset
from .Tokenizer import Tokenizer
from .Generater import Generator as Generator_default
from .Generater import Generator_mo as Generator_mo_default
from .StackTransformer import StackTransformer

from ipdb import set_trace as st

class MoCoTransformer(object):
    def __init__(self, train_opt, model_opt, vqvae_opt, tokenizer_opt, generation_opt, load_vqvae):
        super().__init__()
        self.train_opt = train_opt
        self.model_opt = model_opt
        self.gen_opt = generation_opt
        self.logger = get_logger()
        self.valid_cnt = 0
        self.train_cnt = 0

        # Mask function
        self.gamma = get_gamma_function(train_opt.gamma)
        self.logger.info(f"=== Mask function: {train_opt.gamma} ===")

        # Loss function
        self.loss_func = nn.CrossEntropyLoss()

        # Tokenizer
        self.tokenizer = Tokenizer(vqvae_opt, train_opt, tokenizer_opt, load_vqvae)

        # BiTransformer
        transformer_type = model_opt.get('transformer', 'bi').lower()
        if transformer_type in ['stack']:
            transformer = StackTransformer(T=self.tokenizer.num_frames,
                                           model_opt=model_opt,
                                           vocab_size=self.tokenizer.vocab_size)
        else:
            raise NotImplementedError

        # initialize DeepSpeed
        self.logger.info("*** Total params: {}".format(sum(p.numel() for p in transformer.parameters())))
        parameters = filter(lambda p: p.requires_grad, transformer.parameters())
        model_engine, optimizer, _, _ = deepspeed.initialize(args=train_opt.args,
                                                             model=transformer,
                                                             model_parameters=parameters)

        self.model_engine = model_engine
        self.optimizer = optimizer

        # Load ckpt for transformer & optimizer
        self.start_step, self.start_epoch = 0, 0
        if model_opt.checkpoint_path or model_opt.pretrain_path:
            self.load_ckpt() # load step & epoch
        self.cur_step = self.start_step
        self.cur_epoch = self.start_epoch

    def save_ckpt(self, save_dir):
        ckpt_id = "{:09d}".format(self.cur_step)
        client_sd = {
            'cur_step': self.cur_step,
            'cur_epoch': self.cur_epoch
        }
        self.model_engine.save_checkpoint(save_dir, ckpt_id, client_state=client_sd)

    def load_ckpt(self):
        if self.model_opt.checkpoint_path:
            ckpt_dir = os.path.dirname(self.model_opt.checkpoint_path)
            ckpt_id = os.path.basename(self.model_opt.checkpoint_path)
            if ckpt_id != 'ckpt':
                _, client_sd = self.model_engine.load_checkpoint(ckpt_dir, ckpt_id)
            else:
                _, client_sd = self.model_engine.load_checkpoint(self.model_opt.checkpoint_path)
            self.start_step = client_sd['cur_step']  # Resume training steps
            self.start_epoch = client_sd['cur_epoch']
        else:
            self.start_step = 0
            self.start_epoch = 0
            ckpt_dir = os.path.dirname(self.model_opt.checkpoint_path)
            ckpt_id = os.path.basename(self.model_opt.checkpoint_path)
            if ckpt_id != 'ckpt':
                _, client_sd = self.model_engine.load_checkpoint(ckpt_dir, ckpt_id)
            else:
                _, client_sd = self.model_engine.load_checkpoint(self.model_opt.checkpoint_path)
            _, client_sd = self.model_engine.load_checkpoint(ckpt_dir, ckpt_id)

        self.logger.info("Successfully load ckpt: {} with step {}.".format(_, self.start_step))

    def one_step_forward(self, mo_tar, mo_base, cur_rate):
        # 1. mask suf mo tokens as inputs, fix base mo tokens
        xmo, maskmo = self.tokenizer.random_mask_mo(mo_tar, cur_rate, self.tokenizer.pre_frame_num)
        xmo = torch.cat([mo_base, xmo], dim=1)
        maskmo = torch.cat([torch.zeros_like(mo_base, dtype=torch.int32), maskmo], dim=1)

        # 2. feed the transformer and get logits
        logits_mo = self.model_engine(xmo)  # logits: [B, L, vocab_size]

        # 3. get predicted CE loss for bg/id tokens and masked mo tokens
        target_mo = torch.cat([mo_base, mo_tar], dim=1).reshape(-1)[maskmo.reshape(-1) == 1]
        logit_mo = logits_mo.reshape(-1, logits_mo.shape[2])[maskmo.reshape(-1) == 1, :]
        loss_mo = self.loss_func(logit_mo.float(), target_mo.long())

        loss = loss_mo.to(torch.float16)

        return loss, loss_mo

    def train_one_step(self, inputs):
        '''
            bg1: [B, BGL] bg tokens for the first frame
            id2: [B, IDL] id tokens for the first frame
            bg16: [B, BGL] bg tokens for 16 frames
            id16: [B, IDL] id tokens for 16 frames
            mo16: [B, T * MOL]
        '''
        self.model_engine.train()
        self.valid_cnt = 0

        mo_tar = inputs['mo_tar'].to(self.train_opt.device)
        mo_base = inputs['mo_base'].to(self.train_opt.device)
        cur_rate = self.gamma(np.random.rand(1))

        loss, loss_mo = self.one_step_forward(mo_tar, mo_base, cur_rate)

        # 6. backward & optimize
        self.model_engine.backward(loss)
        self.model_engine.step()
        self.cur_step += 1

        # update wandb info
        if self.cur_step % 10 == 0 and self.train_opt.wandb and dist.get_rank()==0:
            log = {'loss': loss, 'rate': cur_rate,
                   'loss_mo': loss_mo,
                   'learning_rate': self.optimizer.param_groups[0]['lr']}
            wandb.log(log, step=self.cur_step)

        return {'loss': loss}

    def valid_one_step(self, inputs):
        '''
            bg1: [B, BGL] bg tokens for the first frame
            id2: [B, IDL] id tokens for the first frame
            bg16: [B, BGL] bg tokens for 16 frames
            id16: [B, IDL] id tokens for 16 frames
            mo16: [B, T * MOL]
        '''
        self.model_engine.eval()

        mo_tar = inputs['mo_tar'].to(self.train_opt.device)
        mo_base = inputs['mo_base'].to(self.train_opt.device)

        seeds = torch.arange(0.05, 1, 0.05)
        cur_seed = seeds[self.valid_cnt % len(seeds)]
        cur_rate = self.gamma(cur_seed)

        with torch.no_grad():
            loss, loss_mo = self.one_step_forward(mo_tar, mo_base, cur_rate)

        self.valid_cnt += 1
        self.last_valid_input = inputs
        return {'loss': loss, 'loss_mo': loss_mo,
                'loss_bg': torch.tensor(0), 'loss_id': torch.tensor(0)}

    def create_input_tokens_normal(self, mo_base=None):
        # Create blank masked tokens
        if mo_base is not None:
            blank_mo_tokens = torch.ones([self.gen_opt.batch_size,
                                          self.model_opt.mo_token_length - mo_base.shape[1]], dtype=torch.int32)
        else:
            blank_mo_tokens = torch.ones([self.gen_opt.batch_size,
                                          self.model_opt.mo_token_length], dtype=torch.int32)

        masked_mo_tokens = blank_mo_tokens * self.tokenizer.command_tokens['MASK']
        return masked_mo_tokens

    def sample(self, logits):
        """
        logits: B, L, V
        return sample_ids - [B, L]
        """

        b, l, v = logits.shape
        logits_flatten = rearrange(logits, 'b l v -> (b l) v')
        sampled_ids = Categorical(logits=logits_flatten).sample().to(torch.int32)
        sampled_ids = rearrange(sampled_ids, '(b l) -> b l', b=b)
        return sampled_ids

    def generate_one_sample(self, mo_base, timesteps):
        mo_suf_tokens = self.create_input_tokens_normal(mo_base)  # create blank and masked tokens

        # create generator
        mo_generator = Generator_mo_default(num_frames=self.tokenizer.num_frames - self.tokenizer.pre_frame_num,
                                            gen_opt=self.gen_opt, init_tokens=mo_suf_tokens,
                                            start_step=0, tot_steps=timesteps,
                                            mask_token_id=self.tokenizer.command_tokens['MASK'],
                                            gamma_function=self.gamma)


        # Step 2: generate mo tokens
        while mo_generator.unfinished():
            # achieve latest tokens
            mo_suf_tokens = mo_generator.get_cur_tokens()
            # feed the Bitransformer with [wbg, id, mo]
            xmo = torch.cat([mo_base, mo_suf_tokens], dim=1).to(self.train_opt.device)
            with torch.no_grad():
                logits_mo = self.model_engine.forward_mo(xmo)  # logits: [B, L, vocab_size]
            logits_mo = logits_mo[:, mo_base.shape[1]:, self.tokenizer.video_vocab_s:self.tokenizer.cmd_vocab_s]
            # update prediction of mo tokens
            mo_generator.generate_one_step(logits_mo.float().detach().cpu())

        return mo_generator.get_final_tokens() # xbg: [B, L], xid: [B, L], xmo: [B, Step, L]

    def interpolate_one_sample(self, mo_tokens, timesteps):
        """
        bg_tokens: [B, L]
        id_tokens: [B, L]
        mo_tokens: [B, T * L]
        """
        mo_full_tokens = self.create_input_tokens_normal()  # create blank and masked tokens
        mo_full_tokens = self.create_input_tokens_normal()  # create blank and masked tokens
        mo_full_tokens = rearrange(mo_full_tokens, 'B (T L) -> B T L',T=self.tokenizer.num_frames)
        mo_empty_tokens = rearrange(mo_full_tokens[:, :self.tokenizer.num_frames // 2, :].clone(), 'B T L -> B (T L)')
        mo_tokens = rearrange(mo_tokens, 'B (T L) -> B T L', T=self.tokenizer.num_frames)
        for t in range(0, self.tokenizer.num_frames, 2):
            mo_full_tokens[:, t + 1, :] = mo_tokens[:, t // 2, :]

        # create generator
        mo_generator = Generator_mo_default(num_frames=self.tokenizer.num_frames // 2,
                                            gen_opt=self.gen_opt, init_tokens=mo_empty_tokens,
                                            start_step=0, tot_steps=timesteps,
                                            mask_token_id=self.tokenizer.command_tokens['MASK'],
                                            gamma_function=self.gamma)

        # Step 2: generate mo tokens
        while mo_generator.unfinished():
            # achieve latest tokens
            mo_mask_tokens = mo_generator.get_cur_tokens()
            mo_mask_tokens = rearrange(mo_mask_tokens, 'B (T L) -> B T L',
                                       T=self.tokenizer.num_frames // 2)
            for t in range(0, self.tokenizer.num_frames, 2):
                mo_full_tokens[:, t, :] = mo_mask_tokens[:, t // 2, :]
            xmo = rearrange(mo_full_tokens, 'B T L -> B (T L)').to(self.train_opt.device)
            with torch.no_grad():
                logits_mo = self.model_engine.forward_mo(xmo)  # logits: [B, L, vocab_size]
            logits_mo = rearrange(logits_mo, 'B (T L) V -> B T L V', T=self.tokenizer.num_frames)
            logits_mo = rearrange(logits_mo[:, 0::2, :, :], 'B T L V -> B (T L) V')
            logits_mo = logits_mo[:, :, self.tokenizer.video_vocab_s:self.tokenizer.cmd_vocab_s]
            # update prediction of mo tokens
            mo_generator.generate_one_step(logits_mo.float().detach().cpu())

        # ret
        mo_mask_tokens = mo_generator.get_final_tokens()[:, -1, :]
        mo_mask_tokens = rearrange(mo_mask_tokens, 'B (T L) -> B T L',
                                   T=self.tokenizer.num_frames // 2)
        for t in range(0, self.tokenizer.num_frames, 2):
            mo_full_tokens[:, t, :] = mo_mask_tokens[:, t // 2, :]

        return rearrange(mo_full_tokens, 'B T L -> B (T L)') # xbg: [B, L], xid: [B, L], xmo: [B, Step, L]


    def decode_one_clip(self, mo_base, xmo):
        mo_tokens = torch.cat([mo_base, xmo], dim=1).long().to(self.train_opt.device)
        sample_x = self.tokenizer.decode(mo_tokens)  # [B, T, C, H, W]

        tokens = []
        for i in range(mo_tokens.shape[0]):
            token = {'mo_tokens': rearrange(mo_tokens[i].cpu().numpy(), '(t l) -> t l',
                                            t=self.tokenizer.num_frames)}
            tokens.append(token)
        samples = get_seperate_frames(sample_x)
        return samples, tokens

    def generate(self, timesteps=None, mode="test", save_path=None, inputs=None,
                 iterative=1, vp_base_frames=None):
        ''' Video Prediction
            bg1: [B, BGL] bg tokens for the first frame
            id2: [B, IDL] id tokens for the first frame
            bg16: [B, BGL] bg tokens for 16 frames
            id16: [B, IDL] id tokens for 16 frames
            mo16: [B, T * MOL]
        '''
        if inputs is None:
            inputs = self.last_valid_input
        mo_base = inputs['mo_base'][:self.gen_opt.batch_size]
        timesteps = timesteps or self.gen_opt.timesteps
        self.model_engine.eval()

        if iterative == 1:
            xbg, xid, xmo_per_step = self.generate_one_sample(mo_base, timesteps)
        else:
            assert vp_base_frames
            mo_base_length = (self.model_opt.mo_token_length //
                              self.tokenizer.num_frames) * vp_base_frames
            record_mo_base, record_mo_perstep = [], []

            for iter in range(iterative):
                # print(iter)
                xmo_per_step = self.generate_one_sample(mo_base, timesteps)
                record_mo_base.append(mo_base)
                record_mo_perstep.append(xmo_per_step)
                mo_base = xmo_per_step[:, -1, -mo_base_length:]

        # print("Total steps:  ", bg_generator.cur_step)
        if mode == "test":
            assert self.gen_opt.get('show_process', False)
            # visualize all steps
            sample_x = []
            for step in range(timesteps):
                mo_tokens = xmo_per_step[:, step, :].long()
                mo_tokens = torch.cat([mo_base, mo_tokens], dim=1).to(self.train_opt.device)  # [B, L]
                cur_x = self.tokenizer.decode(mo_tokens)
                sample_x.append(cur_x)
            sample_x = torch.cat(sample_x, dim=0) # [B * steps, T, C, H, W]
            show_x = get_visualize_img(sample_x)
            if save_path:
                show_x.save(save_path)
            return show_x

        elif mode == "sample":
            # Only visualize final step
            if iterative == 1:
                samples, tokens = self.decode_one_clip(mo_base, xmo_per_step[:, -1, :])
            else:
                B = self.gen_opt.batch_size
                samples = [[] for b in range(B)]
                for i in range(iterative):
                    # cur_samples: [B, [T]]
                    cur_samples, _ = self.decode_one_clip(record_mo_base[i], record_mo_perstep[i][:, -1, :])
                    if i == 0:
                        samples = [samples[b] + cur_samples[b][:] for b in range(B)]
                    else:
                        samples = [samples[b] + cur_samples[b][-vp_base_frames:] for b in range(B)]
                mo_tokens = torch.cat([item[:, -1, :] for item in record_mo_perstep], dim=0)
                tokens = []
                for i in range(B):
                    token = {'mo_tokens': mo_tokens[i].cpu().numpy()}
                    tokens.append(token)

            return samples, tokens

        else:
            raise NotImplementedError

    def interpolate(self, timesteps=None, mode="test", save_path=None, inputs=None,
                 iterative=1, vp_base_frames=None):
        ''' Video Prediction
            mo16: [B, T * MOL]
        '''
        if inputs is None:
            inputs = self.last_valid_input
        mo_base = inputs['mo_base'][:self.gen_opt.batch_size]
        mo_tar = inputs['mo_tar'][:self.gen_opt.batch_size]
        mo_tokens = torch.cat([mo_base, mo_tar], dim=1)
        timesteps = timesteps or self.gen_opt.timesteps
        self.model_engine.eval()
        xmo_per_step = self.interpolate_one_sample(mo_tokens, timesteps)

        if mode == "sample":
            # Only visualize final step
            samples, tokens = self.decode_one_clip(xmo_per_step[:, :mo_base.shape[1]],
                                                   xmo_per_step[:, mo_base.shape[1]:])
            return samples, tokens

        else:
            raise NotImplementedError
