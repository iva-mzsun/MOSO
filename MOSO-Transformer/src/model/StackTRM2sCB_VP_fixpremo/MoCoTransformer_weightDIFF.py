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
            ckpt_dir = os.path.dirname(self.model_opt.pretrain_path)
            ckpt_id = os.path.basename(self.model_opt.pretrain_path)
            if ckpt_id != 'ckpt':
                _, client_sd = self.model_engine.load_checkpoint(ckpt_dir, ckpt_id)
            else:
                _, client_sd = self.model_engine.load_checkpoint(self.model_opt.pretrain_path)
            _, client_sd = self.model_engine.load_checkpoint(ckpt_dir, ckpt_id)

        self.logger.info("Successfully load ckpt: {} with step {}.".format(_, self.start_step))

    def one_step_forward(self, bg_tar, id_tar, mo_tar,
                         bg_pre, id_pre, mo_base, cur_rate):
        hmo = self.tokenizer.get_hmo(bg_tar.shape[0]).to(self.train_opt.device)

        # 1. mask suf mo tokens as inputs, fix base mo tokens
        xmo, maskmo = self.tokenizer.random_mask_mo(mo_tar, cur_rate, self.tokenizer.pre_frame_num)
        xmo = torch.cat([mo_base, xmo], dim=1)
        maskmo = torch.cat([torch.zeros_like(mo_base, dtype=torch.int32), maskmo], dim=1)

        # 2. feed the transformer and get logits
        logits_bg, logits_id, logits_mo = self.model_engine(bg_pre, id_pre, hmo, xmo)  # logits: [B, L, vocab_size]

        # 3. get predicted CE loss for bg/id tokens and masked mo tokens
        target_bg = bg_tar.reshape(-1).long()
        logit_bg = logits_bg.reshape(-1, logits_bg.shape[2]).to(torch.float32)
        same_mask = bg_pre.reshape(-1) == bg_tar.reshape(-1)
        diff_mask = bg_pre.reshape(-1) != bg_tar.reshape(-1)
        loss_bg_same = self.loss_func(logit_bg[same_mask, :], target_bg[same_mask])
        loss_bg_diff = self.loss_func(logit_bg[diff_mask, :], target_bg[diff_mask])
        loss_bg = 5 * loss_bg_diff + 1 * loss_bg_same

        target_id = id_tar.reshape(-1).long()
        logit_id = logits_id.reshape(-1, logits_id.shape[2]).to(torch.float32)
        same_mask = id_pre.reshape(-1) == id_tar.reshape(-1)
        diff_mask = id_pre.reshape(-1) != id_tar.reshape(-1)
        loss_id_same = self.loss_func(logit_id[same_mask, :], target_id[same_mask])
        loss_id_diff = self.loss_func(logit_id[diff_mask, :], target_id[diff_mask])
        loss_id = 5 * loss_id_diff + 1 * loss_id_same

        target_mo = torch.cat([mo_base, mo_tar], dim=1).reshape(-1)[maskmo.reshape(-1) == 1]
        logit_mo = logits_mo.reshape(-1, logits_mo.shape[2])[maskmo.reshape(-1) == 1, :]
        loss_mo = self.loss_func(logit_mo.float(), target_mo.long())

        loss = ((loss_bg + loss_id + loss_mo) / 3.0).to(torch.float16)

        return loss, loss_bg, loss_id, loss_mo

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

        bg_tar = inputs['bg_tar'].to(self.train_opt.device)
        id_tar = inputs['id_tar'].to(self.train_opt.device)
        mo_tar = inputs['mo_tar'].to(self.train_opt.device)
        bg_pre = inputs['bg_pre'].to(self.train_opt.device)
        id_pre = inputs['id_pre'].to(self.train_opt.device)
        mo_base = inputs['mo_base'].to(self.train_opt.device)
        cur_rate = self.gamma(np.random.rand(1))

        loss, loss_bg, loss_id, loss_mo = self.one_step_forward(bg_tar, id_tar, mo_tar,
                                                                bg_pre, id_pre, mo_base, cur_rate)

        # 6. backward & optimize
        self.model_engine.backward(loss)
        self.model_engine.step()
        self.cur_step += 1

        # update wandb info
        if self.cur_step % 10 == 0 and self.train_opt.wandb and dist.get_rank()==0:
            log = {'loss': loss, 'rate': cur_rate,
                   'loss_mo': loss_mo, 'loss_bg': loss_bg, 'loss_id': loss_id,
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

        bg_tar = inputs['bg_tar'].to(self.train_opt.device)
        id_tar = inputs['id_tar'].to(self.train_opt.device)
        mo_tar = inputs['mo_tar'].to(self.train_opt.device)
        bg_pre = inputs['bg_pre'].to(self.train_opt.device)
        id_pre = inputs['id_pre'].to(self.train_opt.device)
        mo_base = inputs['mo_base'].to(self.train_opt.device)

        seeds = torch.arange(0.05, 1, 0.05)
        cur_seed = seeds[self.valid_cnt % len(seeds)]
        cur_rate = self.gamma(cur_seed)

        with torch.no_grad():
            loss, loss_bg, loss_id, loss_mo = self.one_step_forward(bg_tar, id_tar, mo_tar,
                                                                    bg_pre, id_pre, mo_base, cur_rate)

        self.valid_cnt += 1
        self.last_valid_input = inputs
        return {'loss': loss, 'loss_mo': loss_mo, 'loss_bg': loss_bg, 'loss_id': loss_id}

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

    def generate_one_sample(self, bg_pre, id_pre, mo_base, timesteps):
        mo_suf_tokens = self.create_input_tokens_normal(mo_base)  # create blank and masked tokens
        hmo = self.tokenizer.get_hmo(bg_pre.shape[0]).to(self.train_opt.device)

        # create generator
        mo_generator = Generator_mo_default(num_frames=self.tokenizer.num_frames - self.tokenizer.pre_frame_num,
                                            gen_opt=self.gen_opt, init_tokens=mo_suf_tokens,
                                            start_step=0, tot_steps=timesteps,
                                            mask_token_id=self.tokenizer.command_tokens['MASK'],
                                            gamma_function=self.gamma)

        # Step 1: generate bg/id tokens
        xbg, xid = bg_pre, id_pre
        with torch.no_grad():
            logits_bg, logits_id, stackone_output = self.model_engine.forward_bgid(xbg, xid, hmo)  # logits: [B, L, vocab_size]
        xbg = self.sample(logits_bg[:, :, self.tokenizer.video_vocab_s:self.tokenizer.cmd_vocab_s]).to(
            self.train_opt.device)
        xid = self.sample(logits_id[:, :, self.tokenizer.video_vocab_s:self.tokenizer.cmd_vocab_s]).to(
            self.train_opt.device)

        with torch.no_grad():
            _, _, stackone_output = self.model_engine.forward_bgid(xbg, xid, hmo)
        zmo = stackone_output[:, xbg.shape[1] + xid.shape[1]:, :]  # [B, T, hidden_dim]

        # Step 2: generate mo tokens
        while mo_generator.unfinished():
            # achieve latest tokens
            mo_suf_tokens = mo_generator.get_cur_tokens()
            # feed the Bitransformer with [wbg, id, mo]
            xmo = torch.cat([mo_base, mo_suf_tokens], dim=1).to(self.train_opt.device)
            with torch.no_grad():
                logits_mo = self.model_engine.forward_mo(zmo, xmo)  # logits: [B, L, vocab_size]
            logits_mo = logits_mo[:, mo_base.shape[1]:, self.tokenizer.video_vocab_s:self.tokenizer.cmd_vocab_s]
            # update prediction of mo tokens
            mo_generator.generate_one_step(logits_mo.float().detach().cpu())

        return xbg, xid, mo_generator.get_final_tokens() # xbg: [B, L], xid: [B, L], xmo: [B, Step, L]

    def interpolate_one_sample(self, bg_tokens, id_tokens, mo_tokens, timesteps):
        """
        bg_tokens: [B, L]
        id_tokens: [B, L]
        mo_tokens: [B, T * L]
        """
        hmo = self.tokenizer.get_hmo(mo_tokens.shape[0]).to(self.train_opt.device)

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

        # Step 1: generate bg/id tokens
        with torch.no_grad():
            _, _, stackone_output = self.model_engine.forward_bgid(bg_tokens, id_tokens, hmo)
        zmo = stackone_output[:, bg_tokens.shape[1] + id_tokens.shape[1]:, :]  # [B, T, hidden_dim]

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
                logits_mo = self.model_engine.forward_mo(zmo, xmo)  # logits: [B, L, vocab_size]
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


    def decode_one_clip(self, xbg, xid, mo_base, xmo):
        bg_tokens = xbg.long().to(self.train_opt.device)  # [B, L]
        id_tokens = xid.long().to(self.train_opt.device)  # [B, L]
        mo_tokens = torch.cat([mo_base, xmo], dim=1).long().to(self.train_opt.device)
        sample_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens)  # [B, T, C, H, W]

        tokens = []
        for i in range(bg_tokens.shape[0]):
            token = {'bg_tokens': bg_tokens[i].cpu().numpy(),
                     'id_tokens': id_tokens[i].cpu().numpy(),
                     'mo_tokens': rearrange(mo_tokens[i].cpu().numpy(), '(t l) -> t l',
                                            t=self.tokenizer.num_frames)}
            tokens.append(token)
        samples = get_seperate_frames(sample_x)
        return samples, tokens

    def reencode_last_K_frames(self, xbg, xid, mo_base, xmo, vp_base_frames):
        bg_tokens = xbg.long().to(self.train_opt.device)  # [B, L]
        id_tokens = xid.long().to(self.train_opt.device)  # [B, L]
        mo_tokens = torch.cat([mo_base, xmo], dim=1).long().to(self.train_opt.device)
        sample_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens)  # [B, T, C, H, W]

        B, T, C, H, W = sample_x.shape
        sample_x = torch.clip(sample_x, min=0, max=1)
        sample_x_last = sample_x[:, -vp_base_frames:, :, :, :]
        sample_x_pad = repeat(sample_x[:, -1, :, :, :],
                              'b c h w -> b t c h w', t=T - vp_base_frames)
        input_x = torch.cat([sample_x_last, sample_x_pad], dim=1)
        bg_tokens, id_tokens = self.tokenizer.obtain_bg_id_tokens(input_x)
        return bg_tokens, id_tokens

    def generate(self, timesteps=None, mode="test", save_path=None, inputs=None,
                 iterative=1, vp_base_frames=None, **kwargs):
        ''' Video Prediction
            bg1: [B, BGL] bg tokens for the first frame
            id2: [B, IDL] id tokens for the first frame
            bg16: [B, BGL] bg tokens for 16 frames
            id16: [B, IDL] id tokens for 16 frames
            mo16: [B, T * MOL]
        '''
        if inputs is None:
            inputs = self.last_valid_input
        bg_pre = inputs['bg_pre'][:self.gen_opt.batch_size].to(self.train_opt.device)
        id_pre = inputs['id_pre'][:self.gen_opt.batch_size].to(self.train_opt.device)
        mo_base = inputs['mo_base'][:self.gen_opt.batch_size]
        timesteps = timesteps or self.gen_opt.timesteps
        self.model_engine.eval()

        if iterative == 1:
            xbg, xid, xmo_per_step = self.generate_one_sample(bg_pre, id_pre, mo_base, timesteps)
        else:
            assert vp_base_frames
            mo_base_length = (self.model_opt.mo_token_length //
                              self.tokenizer.num_frames) * vp_base_frames
            record_bg, record_id, record_mo_base, record_mo_perstep = [], [], [], []

            for iter in range(iterative):
                # print(iter)
                xbg, xid, xmo_per_step = self.generate_one_sample(bg_pre, id_pre, mo_base, timesteps)
                record_bg.append(xbg)
                record_id.append(xid)
                record_mo_base.append(mo_base)
                record_mo_perstep.append(xmo_per_step)
                bg_pre, id_pre = xbg, xid # TODO: Original
                # bg_pre, id_pre = self.reencode_last_K_frames(xbg, xid, mo_base, xmo_per_step[:, -1, -mo_base_length:], vp_base_frames)
                mo_base = xmo_per_step[:, -1, -mo_base_length:]

        # print("Total steps:  ", bg_generator.cur_step)
        if mode == "test":
            assert self.gen_opt.get('show_process', False)
            # visualize all steps
            sample_x = []
            for step in range(timesteps):
                bg_tokens = xbg.long().to(self.train_opt.device)  # [B, L]
                id_tokens = xid.long().to(self.train_opt.device)  # [B, L]
                mo_tokens = xmo_per_step[:, step, :].long()
                mo_tokens = torch.cat([mo_base, mo_tokens], dim=1).to(self.train_opt.device)  # [B, L]
                cur_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens)
                sample_x.append(cur_x)
            sample_x = torch.cat(sample_x, dim=0) # [B * steps, T, C, H, W]
            show_x = get_visualize_img(sample_x)
            if save_path:
                show_x.save(save_path)
            return show_x

        elif mode == "sample":
            # Only visualize final step
            if iterative == 1:
                samples, tokens = self.decode_one_clip(xbg, xid, mo_base, xmo_per_step[:, -1, :])
            else:
                B = self.gen_opt.batch_size
                record_mo = []
                samples = [[] for _ in range(B)]
                for i in range(iterative):
                    # cur_samples: [B, [T]]
                    record_mo.append(torch.cat([record_mo_base[i], record_mo_perstep[i][:, -1, :]], dim=1))
                    cur_samples, _ = self.decode_one_clip(record_bg[i], record_id[i],
                                                          record_mo_base[i], record_mo_perstep[i][:, -1, :])
                    if i == 0:
                        samples = [samples[b] + cur_samples[b][:] for b in range(B)]
                    else:
                        samples = [samples[b] + cur_samples[b][vp_base_frames:] for b in range(B)]
                bg_tokens = torch.cat(record_bg, dim=0)
                id_tokens = torch.cat(record_id, dim=0)
                mo_tokens = torch.cat(record_mo, dim=0)
                tokens = []
                for i in range(B):
                    token = {'bg_tokens': bg_tokens[i].cpu().numpy(),
                             'id_tokens': id_tokens[i].cpu().numpy(),
                             'mo_tokens': mo_tokens[i].cpu().numpy()}
                    tokens.append(token)

            return samples, tokens

        else:
            raise NotImplementedError

    def interpolate(self, timesteps=None, mode="test", save_path=None, inputs=None,
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
        bg_tokens = inputs['bg_pre'][:self.gen_opt.batch_size].to(self.train_opt.device)
        id_tokens = inputs['id_pre'][:self.gen_opt.batch_size].to(self.train_opt.device)
        mo_base = inputs['mo_base'][:self.gen_opt.batch_size]
        mo_tar = inputs['mo_tar'][:self.gen_opt.batch_size]
        mo_tokens = torch.cat([mo_base, mo_tar], dim=1)
        timesteps = timesteps or self.gen_opt.timesteps
        self.model_engine.eval()
        xmo_per_step = self.interpolate_one_sample(bg_tokens, id_tokens, mo_tokens, timesteps)

        if mode == "sample":
            # Only visualize final step
            samples, tokens = self.decode_one_clip(bg_tokens, id_tokens,
                                                   xmo_per_step[:, :mo_base.shape[1]],
                                                   xmo_per_step[:, mo_base.shape[1]:])
            return samples, tokens

        else:
            raise NotImplementedError
