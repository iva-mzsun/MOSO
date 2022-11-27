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
from .Generater_woCONFI import Generator_woCONFI
from .Generater_woCONFI import Generator_mo_woCONFI
from .Refiner import ReGenerator, ReGenerator_mo
from .Refiner_woCONFI import Refiner_woCONFI, Refiner_mo_woCONFI
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

        # Judge
        self.if_sep_mo = self.model_opt.get('if_sep_mo', True)
        self.logger.warning(f"******  OPEN {self.if_sep_mo}")

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

    def train_one_step(self, inputs):
        '''
            xbg: [B, bg_seq_L]
            xid: [B, id_seq_L]
            xmo: [B, T * mo_seq_L]
        '''
        xbg = inputs['bg_tokens'].to(self.train_opt.device)
        xid = inputs['id_tokens'].to(self.train_opt.device)
        xmo = inputs['mo_tokens'].to(self.train_opt.device)
        hmo = self.tokenizer.get_hmo(xbg.shape[0]).to(self.train_opt.device)
        self.model_engine.train()
        self.valid_cnt = 0

        # 1. pad vocabs
        gt_bg, gt_id, gt_mo = self.tokenizer.pad_vocab(xbg, xid, xmo)

        # 2. mask tokens as inputs
        cur_rate = self.gamma(np.random.rand(1))
        xbg, maskbg = self.tokenizer.random_mask(gt_bg, cur_rate)
        xid, maskid = self.tokenizer.random_mask(gt_id, cur_rate)
        if self.if_sep_mo is False:
            xmo, maskmo = self.tokenizer.random_mask(gt_mo, cur_rate)
        else:
            xmo, maskmo = self.tokenizer.random_mask_mo(gt_mo, cur_rate)

        # 3. feed the transformer and get logits
        # target = torch.cat([gt_bg, gt_id, gt_mo], dim=1)  # target: [B, L]
        # target_mask = torch.cat([maskbg, maskid, maskmo], dim=1)  # target_mask: [B, L]
        logits_bg, logits_id, logits_mo = self.model_engine(xbg, xid, hmo, xmo)  # logits: [B, L, vocab_size]

        # 4. get predicted CE loss for masked tokens
        target_bg = gt_bg.reshape(-1)[maskbg.reshape(-1) == 1]
        target_id = gt_id.reshape(-1)[maskid.reshape(-1) == 1]
        target_mo = gt_mo.reshape(-1)[maskmo.reshape(-1) == 1]
        logit_bg = logits_bg.reshape(-1, logits_bg.shape[2])[maskbg.reshape(-1) == 1, :]
        logit_id = logits_id.reshape(-1, logits_id.shape[2])[maskid.reshape(-1) == 1, :]
        logit_mo = logits_mo.reshape(-1, logits_mo.shape[2])[maskmo.reshape(-1) == 1, :]

        loss_bg = self.loss_func(logit_bg.float(), target_bg.long()).to(torch.float16)
        loss_id = self.loss_func(logit_id.float(), target_id.long()).to(torch.float16)
        loss_mo = self.loss_func(logit_mo.float(), target_mo.long()).to(torch.float16)
        loss = (loss_bg + loss_id + loss_mo) / 3.0

        # b, l, v = logits.shape
        # target = target.reshape(-1)[target_mask.reshape(-1)==1]
        # logits = logits.reshape(-1, v)[target_mask.reshape(-1)==1, :]
        # loss = self.loss_func(logits.float(), target.long()).to(torch.float16)

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
            xbg: [B, bg_seq_L]
            xid: [B, id_seq_L]
            xmo: [B, T * mo_seq_L]
        '''
        xbg = inputs['bg_tokens'].to(self.train_opt.device)
        xid = inputs['id_tokens'].to(self.train_opt.device)
        xmo = inputs['mo_tokens'].to(self.train_opt.device)
        hmo = self.tokenizer.get_hmo(xbg.shape[0]).to(self.train_opt.device)
        self.model_engine.eval()

        # 1. pad vocabs
        gt_bg, gt_id, gt_mo = self.tokenizer.pad_vocab(xbg, xid, xmo)

        # 2. mask tokens as inputs
        seeds = torch.arange(0.05, 1, 0.05)
        cur_seed = seeds[self.valid_cnt % len(seeds)]
        cur_rate = self.gamma(cur_seed)
        xbg, maskbg = self.tokenizer.random_mask(gt_bg, cur_rate)
        xid, maskid = self.tokenizer.random_mask(gt_id, cur_rate)
        if self.if_sep_mo is False:
            xmo, maskmo = self.tokenizer.random_mask(gt_mo, cur_rate)
        else:
            xmo, maskmo = self.tokenizer.random_mask_mo(gt_mo, cur_rate)

        # 3. feed the transformer and get logits
        # target = torch.cat([gt_bg, gt_id, gt_mo], dim=1)  # target: [B, L]
        # target_mask = torch.cat([maskbg, maskid, maskmo], dim=1)  # target_mask: [B, L]
        with torch.no_grad():
            logits_bg, logits_id, logits_mo = self.model_engine(xbg, xid, hmo, xmo)  # logits: [B, L, vocab_size]

        # 4. get predicted CE loss for masked tokens
        target_bg = gt_bg.reshape(-1)[maskbg.reshape(-1) == 1]
        target_id = gt_id.reshape(-1)[maskid.reshape(-1) == 1]
        target_mo = gt_mo.reshape(-1)[maskmo.reshape(-1) == 1]
        logit_bg = logits_bg.reshape(-1, logits_bg.shape[2])[maskbg.reshape(-1) == 1, :]
        logit_id = logits_id.reshape(-1, logits_id.shape[2])[maskid.reshape(-1) == 1, :]
        logit_mo = logits_mo.reshape(-1, logits_mo.shape[2])[maskmo.reshape(-1) == 1, :]

        loss_bg = self.loss_func(logit_bg.float(), target_bg.long()).to(torch.float16)
        loss_id = self.loss_func(logit_id.float(), target_id.long()).to(torch.float16)
        loss_mo = self.loss_func(logit_mo.float(), target_mo.long()).to(torch.float16)
        loss = (loss_bg + loss_id + loss_mo) / 3.0

        # 4. get predicted CE loss for masked tokens
        # b, l, v = logits.shape
        # target = target.reshape(-1)[target_mask.reshape(-1) == 1]
        # logits = logits.reshape(-1, v)[target_mask.reshape(-1) == 1, :]
        # loss = self.loss_func(logits.float(), target.long()).to(torch.float16)

        self.valid_cnt += 1
        return {'loss': loss, 'loss_mo': loss_mo, 'loss_bg': loss_bg, 'loss_id': loss_id}

    def create_input_tokens_normal(self):
        # Create blank masked tokens
        blank_bg_tokens = torch.ones([self.gen_opt.batch_size, self.model_opt.bg_token_length], dtype=torch.int32)
        blank_id_tokens = torch.ones([self.gen_opt.batch_size, self.model_opt.id_token_length], dtype=torch.int32)
        blank_mo_tokens = torch.ones([self.gen_opt.batch_size, self.model_opt.mo_token_length], dtype=torch.int32)
        masked_bg_tokens = blank_bg_tokens * self.tokenizer.command_tokens['MASK']
        masked_id_tokens = blank_id_tokens * self.tokenizer.command_tokens['MASK']
        masked_mo_tokens = blank_mo_tokens * self.tokenizer.command_tokens['MASK']
        return masked_bg_tokens, masked_id_tokens, masked_mo_tokens

    def generate(self, timesteps=None, mode="test", save_path=None):
        """Unconditional generate"""
        self.model_engine.eval()

        # create blank and masked tokens
        bg_tokens, id_tokens, mo_tokens = self.create_input_tokens_normal()

        generator_type = self.gen_opt.get('name', 'default').lower()
        if generator_type == 'default':
            Generator = Generator_default
            Generator_mo = Generator_mo_default
        elif generator_type == 'woconfidence':
            Generator = Generator_woCONFI
            Generator_mo = Generator_mo_woCONFI
        else:
            raise NotImplementedError

        # create generator
        timesteps = timesteps or self.gen_opt.timesteps
        bg_generator = Generator(gen_opt=self.gen_opt, init_tokens=bg_tokens,
                                 start_step=0, tot_steps=timesteps,
                                 mask_token_id=self.tokenizer.command_tokens['MASK'], gamma_function=self.gamma)
        id_generator = Generator(gen_opt=self.gen_opt, init_tokens=id_tokens,
                                 start_step=0, tot_steps=timesteps,
                                 mask_token_id=self.tokenizer.command_tokens['MASK'], gamma_function=self.gamma)
        mo_generator = Generator_mo(num_frames=self.tokenizer.num_frames,
                                        gen_opt=self.gen_opt, init_tokens=mo_tokens,
                                        start_step=0, tot_steps=timesteps,
                                        mask_token_id=self.tokenizer.command_tokens['MASK'], gamma_function=self.gamma)

        hmo = self.tokenizer.get_hmo(self.gen_opt.batch_size).to(self.train_opt.device)
        while bg_generator.unfinished() \
            and id_generator.unfinished() \
            and mo_generator.unfinished():

            # achieve latest tokens
            bg_tokens = bg_generator.get_cur_tokens()
            id_tokens = id_generator.get_cur_tokens()
            mo_tokens = mo_generator.get_cur_tokens()

            # pad masked tokens
            bg_tokens, id_tokens, mo_tokens = self.tokenizer.pad_vocab(bg_tokens, id_tokens, mo_tokens)

            # feed the Bitransformer with [wbg, id, mo]
            xbg = bg_tokens.to(self.train_opt.device)
            xid = id_tokens.to(self.train_opt.device)
            xmo = mo_tokens.to(self.train_opt.device)
            with torch.no_grad():
                logits_bg, logits_id, logits_mo = self.model_engine(xbg, xid, hmo, xmo) # logits: [B, L, vocab_size]

            # achieve logits
            logits_bg = logits_bg[:, :, self.tokenizer.video_vocab_s:self.tokenizer.cmd_vocab_s].float()
            logits_id = logits_id[:, :, self.tokenizer.video_vocab_s:self.tokenizer.cmd_vocab_s].float()
            logits_mo = logits_mo[:, :, self.tokenizer.video_vocab_s:self.tokenizer.cmd_vocab_s].float()

            # update generators
            bg_generator.generate_one_step(logits_bg.detach().cpu())
            id_generator.generate_one_step(logits_id.detach().cpu())
            mo_generator.generate_one_step(logits_mo.detach().cpu())

        # print("Total steps:  ", bg_generator.cur_step)
        if mode == "test":
            assert self.gen_opt.get('show_process', False)
            # visualize all steps
            sample_x = []
            for step in range(timesteps):
                bg_tokens = bg_generator.get_final_tokens()[:, step, :].long().to(self.train_opt.device)  # [B, L]
                id_tokens = id_generator.get_final_tokens()[:, step, :].long().to(self.train_opt.device)  # [B, L]
                mo_tokens = mo_generator.get_final_tokens()[:, step, :].long().to(self.train_opt.device)  # [B, L]
                cur_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens)
                sample_x.append(cur_x)
            sample_x = torch.cat(sample_x, dim=0) # [B * steps, T, C, H, W]
            show_x = get_visualize_img(sample_x)
            if save_path:
                show_x.save(save_path)
            return show_x

        elif mode == "sample":
            # Only visualize final step
            bg_tokens = bg_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
            id_tokens = id_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
            mo_tokens = mo_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
            sample_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens) # [B, T, C, H, W]

            tokens = []
            for i in range(bg_tokens.shape[0]):
                token = {'bg_tokens': bg_tokens[i].cpu().numpy(),
                         'id_tokens': id_tokens[i].cpu().numpy(),
                         'mo_tokens': rearrange(mo_tokens[i].cpu().numpy(), '(t l) -> t l',
                                                t=self.tokenizer.num_frames)}
                tokens.append(token)
            samples = get_seperate_frames(sample_x)
            return samples, tokens

        else:
            raise NotImplementedError

    def generate_twosteps(self, timesteps=None, mode="test", save_path=None):
        """Unconditional generate"""
        self.model_engine.eval()

        generator_type = self.gen_opt.get('name', 'default').lower()
        if generator_type == 'default':
            Generator = Generator_default
            Generator_mo = Generator_mo_default
        elif generator_type == 'woconfidence':
            Generator = Generator_woCONFI
            Generator_mo = Generator_mo_woCONFI
        else:
            raise NotImplementedError

        # create blank and masked tokens
        bg_tokens, id_tokens, mo_tokens = self.create_input_tokens_normal()

        # create generator
        timesteps = timesteps or self.gen_opt.timesteps
        bg_generator = Generator(gen_opt=self.gen_opt, init_tokens=bg_tokens,
                                 start_step=0, tot_steps=timesteps,
                                 mask_token_id=self.tokenizer.command_tokens['MASK'], gamma_function=self.gamma)
        id_generator = Generator(gen_opt=self.gen_opt, init_tokens=id_tokens,
                                 start_step=0, tot_steps=timesteps,
                                 mask_token_id=self.tokenizer.command_tokens['MASK'], gamma_function=self.gamma)
        mo_generator = Generator_mo(num_frames=self.tokenizer.num_frames,
                                        gen_opt=self.gen_opt, init_tokens=mo_tokens,
                                        start_step=0, tot_steps=timesteps,
                                        mask_token_id=self.tokenizer.command_tokens['MASK'], gamma_function=self.gamma)

        hmo = self.tokenizer.get_hmo(self.gen_opt.batch_size).to(self.train_opt.device)
        # Step 1: generate bg/id tokens
        while bg_generator.unfinished() \
            and id_generator.unfinished():
            # achieve latest tokens
            bg_tokens = bg_generator.get_cur_tokens()
            id_tokens = id_generator.get_cur_tokens()
            # pad masked tokens
            bg_tokens, id_tokens, _ = self.tokenizer.pad_vocab(bg_tokens, id_tokens, None)
            # feed the Bitransformer with [wbg, id, mo]
            xbg = bg_tokens.to(self.train_opt.device)
            xid = id_tokens.to(self.train_opt.device)
            with torch.no_grad():
                logits_bg, logits_id, logits_mo = self.model_engine(xbg, xid, hmo, None) # logits: [B, L, vocab_size]
            # achieve logits
            logits_bg = logits_bg[:, :, self.tokenizer.video_vocab_s:self.tokenizer.cmd_vocab_s].float()
            logits_id = logits_id[:, :, self.tokenizer.video_vocab_s:self.tokenizer.cmd_vocab_s].float()
            # update generators
            bg_generator.generate_one_step(logits_bg.detach().cpu())
            id_generator.generate_one_step(logits_id.detach().cpu())

        # Step 2:
        # prepare bg/id tokens
        bg_tokens = bg_generator.get_cur_tokens()
        id_tokens = id_generator.get_cur_tokens()
        bg_tokens, id_tokens, _ = self.tokenizer.pad_vocab(bg_tokens, id_tokens, None)
        xbg = bg_tokens.to(self.train_opt.device)
        xid = id_tokens.to(self.train_opt.device)
        # generate mo tokens
        while mo_generator.unfinished():
            # achieve latest tokens
            mo_tokens = mo_generator.get_cur_tokens()
            # pad masked tokens
            _, _, mo_tokens = self.tokenizer.pad_vocab(None, None, mo_tokens)
            # feed the Bitransformer with [wbg, id, mo]
            xmo = mo_tokens.to(self.train_opt.device)
            with torch.no_grad():
                _, _, logits_mo = self.model_engine(xbg, xid, hmo, xmo)  # logits: [B, L, vocab_size]
            # update prediction of mo tokens
            logits_mo = logits_mo[:, :, self.tokenizer.video_vocab_s:self.tokenizer.cmd_vocab_s].float()
            mo_generator.generate_one_step(logits_mo.detach().cpu())

        # print("Total steps:  ", bg_generator.cur_step)
        if mode == "test":
            assert self.gen_opt.get('show_process', False)
            # visualize all steps
            sample_x = []
            for step in range(timesteps):
                bg_tokens = bg_generator.get_final_tokens()[:, step, :].long().to(self.train_opt.device)  # [B, L]
                id_tokens = id_generator.get_final_tokens()[:, step, :].long().to(self.train_opt.device)  # [B, L]
                mo_tokens = mo_generator.get_final_tokens()[:, step, :].long().to(self.train_opt.device)  # [B, L]
                cur_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens)
                sample_x.append(cur_x)
            sample_x = torch.cat(sample_x, dim=0) # [B * steps, T, C, H, W]
            show_x = get_visualize_img(sample_x)
            if save_path:
                show_x.save(save_path)
            return show_x

        elif mode == "sample":
            # Only visualize final step
            bg_tokens = bg_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
            id_tokens = id_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
            mo_tokens = mo_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
            sample_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens) # [B, T, C, H, W]

            tokens = []
            for i in range(bg_tokens.shape[0]):
                token = {'bg_tokens': bg_tokens[i].cpu().numpy(),
                         'id_tokens': id_tokens[i].cpu().numpy(),
                         'mo_tokens': rearrange(mo_tokens[i].cpu().numpy(), '(t l) -> t l',
                                                t=self.tokenizer.num_frames)}
                tokens.append(token)
            samples = get_seperate_frames(sample_x)
            return samples, tokens

        else:
            raise NotImplementedError

    def forward(self, bg_tokens, id_tokens, mo_tokens, sample=False):
        """ One forward process
        bg_tokens: [B, BGL]
        id_tokens: [B, IDL]
        mo_tokens: [B, T * MOL]
        return logits
        """
        # pad masked tokens
        hmo = self.tokenizer.get_hmo(self.gen_opt.batch_size).to(self.train_opt.device)
        xbg, xid, xmo = self.tokenizer.pad_vocab(bg_tokens, id_tokens, mo_tokens)

        # feed the Bitransformer with [wbg, id, mo]
        with torch.no_grad():
            logits_bg, logits_id, logits_mo = self.model_engine(xbg, xid, hmo,
                                                                xmo).float()  # logits: [B, L, vocab_size]

        # achieve logits
        logits_bg = logits_bg[:, :, self.tokenizer.video_vocab_s:self.tokenizer.cmd_vocab_s].float()
        logits_id = logits_id[:, :, self.tokenizer.video_vocab_s:self.tokenizer.cmd_vocab_s].float()
        logits_mo = logits_mo[:, :, self.tokenizer.video_vocab_s:self.tokenizer.cmd_vocab_s].float()

        if sample is False:
            return logits_bg, logits_id, logits_mo
        else:
            def get_sample_and_confidence(logits): # logits: [B, L, V], sample_ids/confidence: [B, L]
                logits_flatten = rearrange(logits, 'b l v -> (b l) v')
                sample_ids = Categorical(logits=logits_flatten).sample().to(torch.int32)
                sample_ids = rearrange(sample_ids, '(b l) -> b l', b=logits.shape[0])
                probs = torch.softmax(logits, dim=-1)
                confidence = torch.gather(probs, -1, sample_ids.unsqueeze(-1).to(torch.int64)).squeeze(-1)
                return sample_ids, confidence

            bg_sample, bg_confidence = get_sample_and_confidence(logits_bg)
            id_sample, id_confidence = get_sample_and_confidence(logits_id)
            mo_sample, mo_confidence = get_sample_and_confidence(logits_mo)

            return bg_sample, id_sample, mo_sample, bg_confidence, id_confidence, mo_confidence

    def refine(self, bg_tokens, id_tokens, mo_tokens,
                timesteps=4, cut_prob=0.1,
                mode="test", save_path=None):
        """Refine generated tokens"""
        self.model_engine.eval()

        logits_bg, logits_id, logits_mo = self.forward(bg_tokens, id_tokens, mo_tokens)
        probs_bg = torch.softmax(logits_bg.detach().cpu(), dim=-1)
        probs_id = torch.softmax(logits_id.detach().cpu(), dim=-1)
        probs_mo = torch.softmax(logits_mo.detach().cpu(), dim=-1)
        bg_confidence = torch.gather(probs_bg, -1, bg_tokens.unsqueeze(-1).to(torch.int64)).squeeze(-1)
        id_confidence = torch.gather(probs_id, -1, id_tokens.unsqueeze(-1).to(torch.int64)).squeeze(-1)
        mo_confidence = torch.gather(probs_mo, -1, mo_tokens.unsqueeze(-1).to(torch.int64)).squeeze(-1)

        # create generator
        timesteps = timesteps or self.gen_opt.timesteps
        bg_generator = ReGenerator(gen_opt=self.gen_opt, cut_prob=cut_prob,
                                   start_step=0, tot_steps=timesteps,
                                   init_tokens=bg_tokens, init_confidence=bg_confidence,
                                   mask_token_id=self.tokenizer.command_tokens['MASK'], gamma_function=self.gamma)
        id_generator = ReGenerator(gen_opt=self.gen_opt, cut_prob=cut_prob,
                                   start_step=0, tot_steps=timesteps,
                                   init_tokens=id_tokens, init_confidence=id_confidence,
                                   mask_token_id=self.tokenizer.command_tokens['MASK'], gamma_function=self.gamma)
        if self.if_sep_mo is False:
            mo_generator = ReGenerator(gen_opt=self.gen_opt, cut_prob=cut_prob,
                                       start_step=0, tot_steps=timesteps,
                                       init_tokens=mo_tokens, init_confidence=mo_confidence,
                                       mask_token_id=self.tokenizer.command_tokens['MASK'], gamma_function=self.gamma)
        else:
            mo_generator = ReGenerator_mo(num_frames=self.tokenizer.num_frames,
                                          gen_opt=self.gen_opt, cut_prob=cut_prob,
                                          start_step=0, tot_steps=timesteps,
                                          init_tokens=mo_tokens, init_confidence=mo_confidence,
                                          mask_token_id=self.tokenizer.command_tokens['MASK'], gamma_function=self.gamma)

        while bg_generator.unfinished() \
            and id_generator.unfinished() \
            and mo_generator.unfinished():

            # achieve latest tokens
            bg_tokens = bg_generator.get_cur_tokens()
            id_tokens = id_generator.get_cur_tokens()
            mo_tokens = mo_generator.get_cur_tokens()

            with torch.no_grad():
                logits_bg, logits_id, logits_mo = self.forward(bg_tokens, id_tokens, mo_tokens)

            # update generators
            bg_generator.generate_one_step(logits_bg.detach().cpu())
            id_generator.generate_one_step(logits_id.detach().cpu())
            mo_generator.generate_one_step(logits_mo.detach().cpu())\

            # print(f"*** TO REFRESH BG TOKENS: ", torch.sum(bg_generator.cur_confidence < bg_generator.cut_prob))

        # print("Total steps:  ", bg_generator.cur_step)
        if mode == "test":
            if self.gen_opt.get('show_process', False):
                # visualize all steps
                sample_x = []
                for step in range(timesteps):
                    bg_tokens = bg_generator.get_final_tokens()[:, step, :].long().to(self.train_opt.device)  # [B, L]
                    id_tokens = id_generator.get_final_tokens()[:, step, :].long().to(self.train_opt.device)  # [B, L]
                    mo_tokens = mo_generator.get_final_tokens()[:, step, :].long().to(self.train_opt.device)  # [B, L]
                    cur_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens)
                    sample_x.append(cur_x)
                sample_x = torch.cat(sample_x, dim=0) # [B * steps, T, C, H, W]
                show_x = get_visualize_img(sample_x)
                if save_path:
                    show_x.save(save_path)
                return show_x
            else:
                # Only visualize final step
                bg_tokens = bg_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
                id_tokens = id_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
                mo_tokens = mo_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
                sample_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens)  # [B, T, C, H, W]
                show_x = get_visualize_img(sample_x)
                if save_path:
                    show_x.save(save_path)
                return show_x

        elif mode == "sample":
            # Only visualize final step
            bg_tokens = bg_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
            id_tokens = id_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
            mo_tokens = mo_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
            sample_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens) # [B, T, C, H, W]

            tokens = []
            for i in range(bg_tokens.shape[0]):
                token = {'bg_tokens': bg_tokens[i].cpu().numpy(),
                         'id_tokens': id_tokens[i].cpu().numpy(),
                         'mo_tokens': rearrange(mo_tokens[i].cpu().numpy(), '(t l) -> t l',
                                                t=self.tokenizer.num_frames)}
                tokens.append(token)
            samples = get_seperate_frames(sample_x)
            return samples, tokens

        else:
            raise NotImplementedError

    def refine_twosteps(self, bg_tokens, id_tokens, mo_tokens,
                        timesteps=4, mode="test", save_path=None):
        """Refine generated tokens"""
        self.model_engine.eval()

        # create generator
        timesteps = timesteps or self.gen_opt.timesteps
        bg_generator = Refiner_woCONFI(gen_opt=self.gen_opt,
                                   start_step=0, tot_steps=timesteps,
                                   init_tokens=bg_tokens,
                                   mask_token_id=self.tokenizer.command_tokens['MASK'], gamma_function=self.gamma)
        id_generator = Refiner_woCONFI(gen_opt=self.gen_opt,
                                   start_step=0, tot_steps=timesteps,
                                   init_tokens=id_tokens,
                                   mask_token_id=self.tokenizer.command_tokens['MASK'], gamma_function=self.gamma)
        mo_generator = Refiner_mo_woCONFI(num_frames=self.tokenizer.num_frames,
                                          gen_opt=self.gen_opt,
                                          start_step=0, tot_steps=timesteps,
                                          init_tokens=mo_tokens,
                                          mask_token_id=self.tokenizer.command_tokens['MASK'], gamma_function=self.gamma)
        hmo = self.tokenizer.get_hmo(self.gen_opt.batch_size).to(self.train_opt.device)
        while bg_generator.unfinished() \
            and id_generator.unfinished() \
            and mo_generator.unfinished():

            # achieve latest tokens
            bg_tokens = bg_generator.get_cur_tokens()
            id_tokens = id_generator.get_cur_tokens()
            mo_tokens = mo_generator.get_cur_tokens()

            xbg = bg_tokens.to(self.train_opt.device)
            xid = id_tokens.to(self.train_opt.device)
            xmo = mo_tokens.to(self.train_opt.device)
            with torch.no_grad():
                logits_bg, logits_id, logits_mo = self.model_engine(xbg, xid, hmo, xmo)

            # achieve logits
            logits_bg = logits_bg[:, :, self.tokenizer.bg_vocab_s:self.tokenizer.id_vocab_s].float()
            logits_id = logits_id[:, :, self.tokenizer.id_vocab_s:self.tokenizer.mo_vocab_s].float()
            logits_mo = logits_mo[:, :, self.tokenizer.mo_vocab_s:self.tokenizer.cmd_vocab_s].float()

            # update generators
            bg_generator.generate_one_step(logits_bg.detach().cpu())
            id_generator.generate_one_step(logits_id.detach().cpu())
            mo_generator.generate_one_step(logits_mo.detach().cpu())

            # print(f"*** TO REFRESH BG TOKENS: ", torch.sum(bg_generator.cur_confidence < bg_generator.cut_prob))

        # print("Total steps:  ", bg_generator.cur_step)
        if mode == "test":
            if self.gen_opt.get('show_process', False):
                # visualize all steps
                sample_x = []
                for step in range(timesteps):
                    bg_tokens = bg_generator.get_final_tokens()[:, step, :].long().to(self.train_opt.device)  # [B, L]
                    id_tokens = id_generator.get_final_tokens()[:, step, :].long().to(self.train_opt.device)  # [B, L]
                    mo_tokens = mo_generator.get_final_tokens()[:, step, :].long().to(self.train_opt.device)  # [B, L]
                    cur_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens)
                    sample_x.append(cur_x)
                sample_x = torch.cat(sample_x, dim=0) # [B * steps, T, C, H, W]
                show_x = get_visualize_img(sample_x)
                if save_path:
                    show_x.save(save_path)
                return show_x
            else:
                # Only visualize final step
                bg_tokens = bg_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
                id_tokens = id_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
                mo_tokens = mo_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
                sample_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens)  # [B, T, C, H, W]
                show_x = get_visualize_img(sample_x)
                if save_path:
                    show_x.save(save_path)
                return show_x

        elif mode == "sample":
            # Only visualize final step
            bg_tokens = bg_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
            id_tokens = id_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
            mo_tokens = mo_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
            sample_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens) # [B, T, C, H, W]

            tokens = []
            for i in range(bg_tokens.shape[0]):
                token = {'bg_tokens': bg_tokens[i].cpu().numpy(),
                         'id_tokens': id_tokens[i].cpu().numpy(),
                         'mo_tokens': rearrange(mo_tokens[i].cpu().numpy(), '(t l) -> t l',
                                                t=self.tokenizer.num_frames)}
                tokens.append(token)
            samples = get_seperate_frames(sample_x)
            return samples, tokens

        else:
            raise NotImplementedError


    def refine0(self, bg_tokens, id_tokens, mo_tokens,
               timesteps=4, cut_prob=0.1,
               mode="test", save_path=None):
        """Refine generated tokens"""
        self.model_engine.eval()

        # create generator
        timesteps = timesteps or self.gen_opt.timesteps
        bg_generator = ReGenerator(gen_opt=self.gen_opt, init_tokens=bg_tokens,
                                   start_step=0, tot_steps=timesteps, cut_prob=cut_prob)
        id_generator = ReGenerator(gen_opt=self.gen_opt, init_tokens=id_tokens,
                                   start_step=0, tot_steps=timesteps, cut_prob=cut_prob)
        if self.if_sep_mo is False:
            mo_generator = ReGenerator(gen_opt=self.gen_opt, init_tokens=mo_tokens,
                                       start_step=0, tot_steps=timesteps, cut_prob=cut_prob)
        else:
            mo_generator = ReGenerator_mo(num_frames=self.tokenizer.num_frames,
                                          gen_opt=self.gen_opt, init_tokens=mo_tokens,
                                          start_step=0, tot_steps=timesteps, cut_prob=cut_prob)

        while bg_generator.unfinished() \
            and id_generator.unfinished() \
            and mo_generator.unfinished():

            # achieve latest tokens
            bg_tokens = bg_generator.get_cur_tokens()
            id_tokens = id_generator.get_cur_tokens()
            mo_tokens = mo_generator.get_cur_tokens()

            # pad masked tokens
            bg_tokens, id_tokens, mo_tokens = self.tokenizer.pad_vocab(bg_tokens, id_tokens, mo_tokens)

            # feed the Bitransformer with [wbg, id, mo]
            with torch.no_grad():
                logits = self.model_engine(
                    torch.cat([bg_tokens, id_tokens, mo_tokens], dim=1).to(self.train_opt.device)).float()  # logits: [B, L, vocab_size]

            # achieve logits
            logits_bg = logits[:, :bg_tokens.shape[1],
                        self.tokenizer.bg_vocab_s:self.tokenizer.id_vocab_s]
            logits_id = logits[:, bg_tokens.shape[1]:bg_tokens.shape[1] + id_tokens.shape[1],
                        self.tokenizer.id_vocab_s:self.tokenizer.mo_vocab_s]
            logits_mo = logits[:, bg_tokens.shape[1] + id_tokens.shape[1]:,
                        self.tokenizer.mo_vocab_s:self.tokenizer.cmd_vocab_s]

            # update generators
            bg_generator.generate_one_step(logits_bg.detach().cpu())
            id_generator.generate_one_step(logits_id.detach().cpu())
            mo_generator.generate_one_step(logits_mo.detach().cpu())

        # print("Total steps:  ", bg_generator.cur_step)
        if mode == "test":
            if self.gen_opt.get('show_process', False):
                # visualize all steps
                sample_x = []
                for step in range(timesteps):
                    bg_tokens = bg_generator.get_final_tokens()[:, step, :].long().to(self.train_opt.device)  # [B, L]
                    id_tokens = id_generator.get_final_tokens()[:, step, :].long().to(self.train_opt.device)  # [B, L]
                    mo_tokens = mo_generator.get_final_tokens()[:, step, :].long().to(self.train_opt.device)  # [B, L]
                    cur_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens)
                    sample_x.append(cur_x)
                sample_x = torch.cat(sample_x, dim=0) # [B * steps, T, C, H, W]
                show_x = get_visualize_img(sample_x)
                if save_path:
                    show_x.save(save_path)
                return show_x
            else:
                # Only visualize final step
                bg_tokens = bg_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
                id_tokens = id_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
                mo_tokens = mo_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
                sample_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens)  # [B, T, C, H, W]
                show_x = get_visualize_img(sample_x)
                if save_path:
                    show_x.save(save_path)
                return show_x

        elif mode == "sample":
            # Only visualize final step
            bg_tokens = bg_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
            id_tokens = id_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
            mo_tokens = mo_generator.get_final_tokens()[:, -1, :].long().to(self.train_opt.device)  # [B, L]
            sample_x = self.tokenizer.decode(bg_tokens, id_tokens, mo_tokens) # [B, T, C, H, W]

            tokens = []
            for i in range(bg_tokens.shape[0]):
                token = {'bg_tokens': bg_tokens[i].cpu().numpy(),
                         'id_tokens': id_tokens[i].cpu().numpy(),
                         'mo_tokens': rearrange(mo_tokens[i].cpu().numpy(), '(t l) -> t l',
                                                t=self.tokenizer.num_frames)}
                tokens.append(token)
            samples = get_seperate_frames(sample_x)
            return samples, tokens

        else:
            raise NotImplementedError