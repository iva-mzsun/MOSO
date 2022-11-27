import os
import time
import math
import json
import wandb
import deepspeed
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist

from src.utils import  *
from src.model import get_model
from src.dataset import get_dataloader

from ipdb import set_trace as st

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="MoCo")
        parser.add_argument('--opt', required=True, type=str,
                            help="path of the config file")
        parser.add_argument('--local_rank', type=int)
        # Include DeepSpeed configuration arguments
        parser = deepspeed.add_config_arguments(parser)
        args = parser.parse_args()
        # global config
        opt = get_obj_from_yaml(args.opt)
        opt.train.args = args
        opt.train.local_rank = args.local_rank
        opt.train.device = torch.device('cuda', args.local_rank)
        try:
            opt.train.batch_size = json.load(open(opt.train.deepspeed_config, 'r'))['train_batch_size']
        except:
            opt.train.batch_size = json.load(open(opt.train.deepspeed_config, 'r'))['train_micro_batch_size_per_gpu']

        # config of exp_name and exp_path
        if opt.model.checkpoint_path:
            assert os.path.exists(opt.model.checkpoint_path)
            ckpt_dir = os.path.dirname(opt.model.checkpoint_path)
            ckpt_id = os.path.basename(opt.model.checkpoint_path)
            if ckpt_id != 'ckpt':
                opt.train.exp_name = os.path.basename(os.path.dirname(ckpt_dir))
                opt.train.exp_path = os.path.abspath(os.path.dirname(ckpt_dir))
            else:
                opt.train.exp_name = os.path.basename(os.path.dirname(opt.model.checkpoint_path))
                opt.train.exp_path = os.path.abspath(os.path.dirname(opt.model.checkpoint_path))
        else:
            assert opt.train.exp_name
            # TODO: to recover
            opt.train.exp_name = opt.train.exp_name + time.strftime("_%Y%m%d-%H%M%S", time.localtime())
            opt.train.exp_path = os.path.join('experiments', opt.train.exp_name)
            os.makedirs(opt.train.exp_path, exist_ok=True)
            os.makedirs(os.path.join(opt.train.exp_path, 'gen'), exist_ok=True)
            os.makedirs(os.path.join(opt.train.exp_path, 'ckpt'), exist_ok=True)

        # initialize distributed config
        deepspeed.init_distributed()
        setup_seed(0)

        # set logger
        if dist.get_rank() == 0:
            logger = get_logger(os.path.join(opt.train.exp_path, 'logging-{}.log'.
                                             format(time.strftime("_%Y%m%d-%H%M%S", time.localtime()))),
                                opt.train.exp_name, isopen=True)
            # print config infos
            logger.info(f"** Load OPT file: {args.opt}")
            logger.info(f"** Experiment information is saved in {opt.train.exp_path}")
            logger.info(f"** configs:")
            print_opts(opt, logger)

            # initialize wandb
            project_name = opt.train.exp_name + time.strftime("_FOLLOW%M%S", time.localtime()) \
                            if opt.model.checkpoint_path else opt.train.exp_name
            if opt.train.wandb:
                wandb.init(project='MoCo', notes=str(opt),
                           group=opt.model.name.lower(),
                           tags=[opt.model.name.lower(),
                                  opt.dataset.name.lower()],
                           entity="mzsun", id=project_name, resume='allow')
        else:
            logger = get_logger(logging_file=None, name=None, isopen=False)


        # load data
        logger.info(f"Start to load data...")
        trainloader, trainsampler, validloader, validsampler = get_dataloader(opt)
        logger.info(f"** Total train iterations: {opt.train.num_train_steps}")
        logger.info(f"** Expect total epochs: {opt.train.num_train_steps / len(trainloader)}")

        # load model, Do Not Load VQVAE when training
        model, start_step, start_epoch = get_model(opt, load_vqvae=True)

        # start training
        val_loss = -1
        timer = get_timer()
        timer.start('train', pad=start_step)
        logger.info(f"Start training...")

        while model.cur_step < opt.train.num_train_steps:
            trainsampler.set_epoch(model.cur_epoch)
            # validsampler.set_epoch(model.cur_epoch)

            for _, inputs in enumerate(trainloader):

                output = model.train_one_step(inputs)
                # print train infos
                if model.cur_step % 50 == 0:
                    left_t, expect_t = timer.duration('train', model.cur_step, opt.train.num_train_steps)
                    cur_lr = model.optimizer.param_groups[0]['lr']
                    logger.info("{}/{}/{} lr:{:.2f} loss:{:.6f} previous_val:{:.2f} left_T:{} expect_T:{}".format(
                        model.cur_epoch, model.cur_step, opt.train.num_train_steps,
                        cur_lr, output['loss'], val_loss, left_t, expect_t
                    ))

                # save ckpt
                if model.cur_step % opt.train.save_ckpt_steps == 0:
                    model.save_ckpt(save_dir=os.path.join(opt.train.exp_path, 'ckpt'))

                # validation
                if model.cur_step % opt.train.num_valid_steps == 0:
                    logger.info(f"====================validation start - {model.cur_step}========================")
                    timer.start('valid')
                    val_loss, val_loss_bg, val_loss_id, val_loss_mo = [], [], [], []
                    for _, val_inputs in enumerate(validloader):
                        output = model.valid_one_step(val_inputs)
                        val_loss.append(output['loss'].detach().cpu().numpy())
                        val_loss_bg.append(output['loss_bg'].detach().cpu().numpy())
                        val_loss_id.append(output['loss_id'].detach().cpu().numpy())
                        val_loss_mo.append(output['loss_mo'].detach().cpu().numpy())
                        if (_ + 1) % 100 == 0:
                            logger.info(f"{_ + 1}/{len(validloader)}  valid_loss: {np.mean(val_loss)}")
                    val_loss = np.mean(val_loss)
                    if opt.train.wandb and dist.get_rank() == 0:
                        wandb.log({'valid_loss': np.mean(val_loss),
                                   'valid_loss_bg': np.mean(val_loss_bg),
                                   'valid_loss_id': np.mean(val_loss_id),
                                   'valid_loss_mo': np.mean(val_loss_mo)}, step=model.cur_step)
                    logger.info(f"validation finished with val_loss: {np.mean(val_loss)} after {timer.end('valid', mode='m')}.")
                    logger.info(f"============================validation  end================================")

                    logger.info(f"====================generation start - {model.cur_step}========================")
                    timer.start('gen')
                    if dist.get_rank() == 0:
                        save_path = os.path.join(opt.train.exp_path, 'gen', f"STEP{model.cur_step}.png")
                        show_x = model.generate(save_path=save_path)
                        logger.info(f"Save generated image to {save_path}.")
                        if opt.train.wandb:
                            show_x = wandb.Image(show_x, caption=f"valid x of {model.cur_step}")
                            wandb.log({'generate_img': show_x}, step=model.cur_step)
                    logger.info(f"=============generation end after {timer.end('gen', mode='s')}=================")

                # finish training
                if model.cur_step >= opt.train.num_train_steps:
                    break

            model.cur_epoch += 1









