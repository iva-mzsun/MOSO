import ipdb
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import argparse
import lpips
from einops import rearrange, repeat
from .VQ_EMA import VectorQuantizerEMA
from .Encoder_Motion import Encoder_Motion, Encoder_Motion_TA
from .Encoder_Identity import Encoder_Identity
from .Encoder_Background import Encoder_Background
from .Decoder import Decoder
from .IMG_Discriminator import NLayerDiscriminator
try:
    from src.losses.GAN_loss import hinge_d_loss
except:
    from MoCoVQVAE.src.losses.GAN_loss import hinge_d_loss
from pytorch_msssim import ssim

class VQVAEModel(nn.Module):
    def __init__(self, model_opt, opt):
        super(VQVAEModel, self).__init__()

        # some parameters
        num_hiddens = model_opt['num_hiddens']
        num_residual_layers = model_opt['num_residual_layers']
        num_residual_hiddens= model_opt['num_residual_hiddens']
        suf_method = model_opt['suf_method']
        ds_motion = model_opt['ds_motion']
        ds_identity = model_opt['ds_identity']
        ds_background = model_opt['ds_background']
        num_frames = opt['dataset']['num_frames']
        num_head = model_opt['num_head']
        embedding_dim = model_opt['embedding_dim']
        num_embeddings = model_opt['num_embeddings']
        commitment_cost = model_opt['commitment_cost']
        decay = model_opt['decay']
        augcb = model_opt['if_augcb']
        ds_content = model_opt['ds_content']

        time_head = model_opt.get('time_head', opt['dataset']['num_frames'])
        print(f"!!!!!!!!!!! time head: {time_head} !!!!!!!!!!!!!")

        # discriminator
        self._disc_start_step = opt['train']['disc_start_step']
        self._discriminator = NLayerDiscriminator(**model_opt['disc_opt'])
        self._discriminator_loss = hinge_d_loss

        # get LPIPS loss
        self.perceptual_factor = model_opt['lpips_factor']
        self.perceptual_loss = None

        # weights for ABS/MSE recon loss and Generator loss
        self.generator_weight = model_opt['Gen_weight']

        # Construct model
        self._encoder_bg = Encoder_Background(ds_content=ds_background,
                                              num_hiddens=num_hiddens,
                                              num_residual_layers=num_residual_layers,
                                              num_residual_hiddens=num_residual_hiddens,
                                              T=num_frames, suf_method=suf_method)


        if model_opt['encoder_mo_type'] is None or model_opt['encoder_mo_type'] == 'default':
            print(f"Loading Motion Encoder: Encoder_Motion...")
            self._encoder_mo = Encoder_Motion(ds_motion=ds_motion,
                                              num_hiddens=num_hiddens,
                                              num_residual_layers=num_residual_layers,
                                              num_residual_hiddens=num_residual_hiddens,
                                              n_head=num_head, d_model=num_hiddens, d_kv=64,
                                              time_head=time_head)
        elif model_opt['encoder_mo_type'].lower() == 'time-agnostic':
            print(f"Loading Motion Encoder: Encoder_Motion_TA...")
            self._encoder_mo = Encoder_Motion_TA(ds_motion=ds_motion,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens,
                                                 n_head=num_head, d_model=num_hiddens, d_kv=64)
        else:
            raise ValueError(f"No implemention for encoder_mo_type: {model_opt['encoder_mo_type']}.")


        if model_opt.get('decoder_type', 'default') in ['default', 'decoder_woPA']:
            self._decoder = Decoder(num_hiddens=num_hiddens,
                                    num_residual_layers=ds_content,
                                    num_residual_hiddens=num_residual_hiddens,
                                    ds_content=ds_content,
                                    ds_motion=ds_motion,
                                    ds_identity=ds_background,
                                    ds_background=ds_background)
        else:
            raise ValueError(f"No implemention for decoder_type: {model_opt.get('decoder_type', 'default')}.")

        self._pre_vq_bg = nn.Conv2d(num_hiddens, embedding_dim,
                                 kernel_size=1, stride=1, padding=0)
        self._suf_vq_bg = nn.ConvTranspose2d(embedding_dim, num_hiddens,
                                          kernel_size=1, stride=1, padding=0)
        self._pre_vq_mo = nn.Conv2d(num_hiddens, embedding_dim,
                                 kernel_size=1, stride=1, padding=0)
        self._suf_vq_mo = nn.ConvTranspose2d(embedding_dim, num_hiddens,
                                          kernel_size=1, stride=1, padding=0)

        self._vq_ema = VectorQuantizerEMA(embedding_dim=embedding_dim,
                                          num_embeddings=num_embeddings,
                                          commitment_cost=commitment_cost,
                                          decay=decay, if_augcb=augcb)

        self._data_variance = 0.0632704

    def _decode(self, bg_tokens, id_tokens, mo_tokens):
        '''
            bg_tokens: [B, 1, H, W]
            id_tokens: [B, 1, H, W]
            mo_tokens: [B, T, H, W]
        '''
        B = bg_tokens.shape[0]
        vq_bg = self._vq_ema.quantize_code(bg_tokens)
        vq_mo = self._vq_ema.quantize_code(mo_tokens)

        quantize_bg = self._suf_vq_bg(vq_bg)
        quantize_mo = self._suf_vq_mo(vq_mo)

        quantize_bg = rearrange(quantize_bg, "(b t) c h w -> b t c h w", b=B)
        quantize_mo = rearrange(quantize_mo, "(b t) c h w -> b t c h w", b=B)

        # get recon loss
        x_rec, _, _ = self._decoder(quantize_bg, quantize_bg, quantize_mo)
        return x_rec

    def _generater(self, batch, is_training):
        """
        MoCoVQVAE
        x: [B, T, C, H, W]
        xc: [B, 1, D, H', W']
        xm: [B, T, D, H', W']
        """
        x, xbg, xid, xmo = batch
        B, _, _, _, _ = xbg.shape
        feat_bg = self._encoder_bg(xbg)
        feat_mo = self._encoder_mo(xmo)

        feat_bg = rearrange(feat_bg, "b t c h w -> (b t) c h w")
        feat_mo = rearrange(feat_mo, "b t c h w -> (b t) c h w")

        feat_bg = self._pre_vq_bg(feat_bg)
        feat_mo = self._pre_vq_mo(feat_mo)

        vq_output_bg = self._vq_ema(feat_bg, is_training=is_training)
        vq_output_mo = self._vq_ema(feat_mo, is_training=is_training)

        quantize_bg = self._suf_vq_bg(vq_output_bg['quantize'])
        quantize_mo = self._suf_vq_mo(vq_output_mo['quantize'])

        quantize_bg = rearrange(quantize_bg, "(b t) c h w -> b t c h w", b=B)
        quantize_mo = rearrange(quantize_mo, "(b t) c h w -> b t c h w", b=B)

        # get recon loss
        x_rec, _, _ = self._decoder(quantize_bg, quantize_bg, quantize_mo)
        abs_loss = torch.abs(x_rec - x).mean()
        mse_loss = torch.tensor(0.0, dtype=torch.float32)
        recon_loss =  abs_loss

        # SSIM loss
        tx = rearrange(x, 'b t c h w -> (b t) c h w')
        tx_rec = rearrange(x_rec, 'b t c h w -> (b t) c h w')
        with torch.no_grad():
            ssim_val = ssim(tx, tx_rec, data_range=1, size_average=True)

        # LPIPS loss
        if self.perceptual_loss is None:
            self.perceptual_loss = lpips.LPIPS(net='vgg', pnet_tune=False).to(x.device)
        p_loss = self.perceptual_loss(tx * 2 - 1, tx_rec * 2 - 1).mean()
        nll_loss = recon_loss + p_loss * self.perceptual_factor

        # Total Loss
        loss = nll_loss + vq_output_bg['loss'] + vq_output_mo['loss']

        return {
            'loss': loss,
            'nll_loss': nll_loss,
            'x_rec': x_rec,
            'quantize_bg': quantize_bg,
            'quantize_id': quantize_bg,
            'quantize_mo': quantize_mo,
            'vq_output_bg': vq_output_bg,
            'vq_output_id': vq_output_bg,
            'vq_output_mo': vq_output_mo,
            'record_logs':{
                'ssim_val': ssim_val,
                'abs_loss': abs_loss,
                'mse_loss': mse_loss,
                'rec_loss': recon_loss,
                'lpips_loss': p_loss,
                'quant_loss_bg': vq_output_bg['loss'],
                'quant_loss_id': vq_output_bg['loss'],
                'quant_loss_mo': vq_output_mo['loss'],
            }
        }

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.1, 1e4).detach()
        return d_weight.detach()

    def forward_step(self, inputs, is_training, optimizer_idx):
        if optimizer_idx <= 0: # train generator
            record_logs = {}
            logs = self._generater(inputs, is_training=is_training)
            loss, nll_loss, x_rec = logs['loss'], logs['nll_loss'], logs['x_rec']
            record_logs.update(logs['record_logs'])

            if optimizer_idx == 0:
                # get discriminator judges, TODO: To Zero grads in train!
                logits_fake = self._discriminator(x_rec.contiguous())
                gan_loss = -torch.mean(logits_fake)
                if self.generator_weight is None:
                    generator_weight = self.calculate_adaptive_weight(nll_loss, gan_loss,
                                                                  last_layer=self._decoder._last_layer.weight)
                else:
                    generator_weight = self.generator_weight
                record_logs.update({
                    'G0_generator_loss': gan_loss,
                    'G0_0_generator_weight': generator_weight
                })
            elif optimizer_idx == -1:
                gan_loss = torch.tensor(0.0)
                generator_weight = torch.tensor(0.0)
            else:
                raise ValueError(f"No implemention for optimizer_idx: {optimizer_idx}.")

            ret_loss = loss + generator_weight * gan_loss

            return {
                'loss': ret_loss,
                'x_rec': x_rec,
                'quantize_bg': logs['quantize_bg'],
                'quantize_id': logs['quantize_id'],
                'quantize_mo': logs['quantize_mo'],
                'ssim_metric': logs['record_logs']['ssim_val'],
                'rec_loss': logs['record_logs']['rec_loss'],
                'lpips_loss': logs['record_logs']['lpips_loss'],
                'record_logs': record_logs
            }

        elif optimizer_idx == 1:
            with torch.no_grad():
                output = self._generater(inputs, is_training=False)
            x, _, _, _ = inputs
            x_rec = output['x_rec']
            logits_real = self._discriminator(x.contiguous())
            logits_fake = self._discriminator(x_rec.contiguous())

            gan_loss = self._discriminator_loss(logits_real, logits_fake)
            return {
                "loss": gan_loss,
                "record_logs": {
                    "G1_discriminator_loss": gan_loss.clone().detach().mean(),
                    "G1_0_logits_real": logits_real.detach().mean(),
                    "G1_1_logits_fake": logits_fake.detach().mean()
                }
            }

        else:
            raise ValueError(f"No implemention for optimizer_idx: {optimizer_idx}.")

    def forward(self, inputs, is_training,
                optimizer=None, iteration=None, wandb_open=False, writer=None):
        logs = {}
        if iteration is None:
            assert is_training is False
            optimizer_idx = -1 # Not include discriminator
        elif iteration < self._disc_start_step:
            optimizer_idx = -1 # Not include discriminator
        else:
            optimizer_idx = iteration % 2 # 0: Generator, 1: Discriminator

        output = self.forward_step(inputs, is_training, optimizer_idx=optimizer_idx)

        if (dist.is_initialized() is False or dist.get_rank() == 0) and (is_training and iteration % 9 == 0):
            logs.update({"learning_rate": optimizer.state_dict()['param_groups'][0]['lr']})
            logs.update(output['record_logs'])
            if is_training == True and wandb_open == True:
                wandb.log(logs, step=iteration)

            if is_training == True and writer is not None:
                if optimizer_idx <= 0:
                    writer.add_scalar('train/rec_loss: ', logs['rec_loss'], iteration)
                    writer.add_scalar('train/quant_loss_bg_loss: ', logs['quant_loss_bg'], iteration)
                    writer.add_scalar('train/quant_loss_id_loss: ', logs['quant_loss_id'], iteration)
                    writer.add_scalar('train/quant_loss_mo_loss: ', logs['quant_loss_mo'], iteration)
                    writer.add_scalar('train/learning_rate: ', logs['learning_rate'], iteration)
                    if 'G0_generator_loss' in logs.keys():
                        writer.add_scalar('train/G0_generator_loss: ', logs['G0_generator_loss'], iteration)
                        writer.add_scalar('train/G0_0_generator_weight: ', logs['G0_0_generator_weight'], iteration)
                else:
                    writer.add_scalar('train/G1_discriminator_loss: ', logs['G1_discriminator_loss'], iteration)

        output['optimizer_idx'] = optimizer_idx
        return output


def printkey(keys):
    for item in keys:
        print(item)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQVAE2")
    parser.add_argument('--downsample', default=8, type=int)
    args = parser.parse_args(args=[])

    model = VQVAEModel(64, 32, 2, 128, 64, 0.25, 0.99, args)

    # model = Decoder(64, 2, 32, 4)
    # summary(model, input_size=(3, 256, 256))

    # printkey(model.state_dict().keys())

    # input = torch.randn([32, 3, 256, 256])
    # flops, params = profile(model, inputs=(input,))
    # print(flops)
    # print(params)

    # params = torch.sum
