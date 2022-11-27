import os
os.environ['KMP_WARNINGS'] = 'off'
import time
import math
import argparse
import numpy as np
from thop import profile
from PIL import Image

import torch
import torch.utils.data as Data
import torch.nn as nn
from torchsummary import summary
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from src.model import get_model
from src.dataset import get_dataset
from src.utils import get_logger

from ipdb import set_trace as st
def debug():
    if dist.get_rank() == 0:
        st()
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()

# set random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_opt_from_yaml(path):
    assert os.path.exists(path), f"{path} must exists!"
    import yaml
    with open(path, 'r') as f:
        opt = yaml.load(f)
    return opt

def get_data_loaders(opt):
    data_opt = opt['dataset']
    Logger = get_logger()
    Logger.info(f"Start to get dataset {data_opt['name']}...")
    trainset, testset = get_dataset(opt)

    train_sampler = Data.distributed.DistributedSampler(trainset)
    trainloader = Data.DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=data_opt['batch_size'],
        num_workers=data_opt['num_workers'],
        pin_memory=data_opt['pin_memory'],
        shuffle=data_opt['shuffle'],
        drop_last=True)

    if data_opt['val'] is not None:
        Logger.info("testset is Valid.")
        test_sampler = Data.distributed.DistributedSampler(testset)
        testloader = Data.DataLoader(
            testset,
            sampler=test_sampler,
            batch_size=data_opt['batch_size'],
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory'],
            shuffle=data_opt['shuffle'],
            drop_last=True)
    else:
        Logger.info("testset is NONE.")
        testloader = None
        test_sampler = None

    return trainloader, testloader, train_sampler, test_sampler

def print_opts(opt, logger, start=2):
    if isinstance(opt, dict):
        for key, value in opt.items():
            if isinstance(value, dict):
                logger.info(' '*start + str(key))
                print_opts(value, logger, start+4)
            else:
                logger.info(' ' * start + str(key) + ' ' * start + str(value))
    else:
        logger.info(' '*start + str(opt))

def get_visualize_img(img): # img: [B T C H W]
    # mean = torch.tensor([0.485, 0.456, 0.406])
    # std = torch.tensor([0.229, 0.224, 0.225])
    # x = img[:8].detach().cpu() * std[None, None, :, None, None] + \
    #     mean[None, None, :, None, None]
    x = img[:8].detach().cpu()
    show_x = torch.clamp(x, min=0, max=1)
    b, t, c, h, w = show_x.shape
    show_x = show_x.permute((0, 3, 1, 4, 2)).numpy()
    show_x = show_x.reshape((b * h, t * w, c)) * 255.
    show_x = Image.fromarray(show_x.astype(np.uint8)).convert('RGB')
    return show_x

def validsample(model, val_inputs, train_opt):
    try:
        val_inputs = [val_inputs['ret_img'], val_inputs['ret_img'], val_inputs['ret_img'], val_inputs['ret_img_mo']]
    except:
        xfull, xbg, xid, xmo = val_inputs
        val_inputs = [xfull, xfull, xfull, xmo]

    val_inputs = [item.to(train_opt['device']) for item in val_inputs]
    output = model(val_inputs, is_training=False, writer=None)
    cur_ssim = output['ssim_metric'].clone().detach().cpu().numpy() if 'ssim_metric' in output.keys() else 0
    # cur_msssim = output['msssim_metric'].clone().detach().cpu().numpy() if 'msssim_metric' in output.keys() else 0
    cur_rec = output['rec_loss'].clone().detach().cpu().numpy() if 'rec_loss' in output.keys() else 0
    cur_lpips = output['lpips_loss'].clone().detach().cpu().numpy() if 'lpips_loss' in output.keys() else 0
    print(f"SSIM: {cur_ssim}    MSE:  {cur_rec}    LPIPS: {cur_lpips}")

def validation(model, testloader, train_opt, iteration):
    model.eval()
    val_time = time.time()
    Logger = get_logger()
    testloader_iter = enumerate(testloader)
    fullbgid = train_opt.get('fullbgid_start', train_opt['disc_start_step'])

    rec_loss, ssim_metric, msssim_metric, lpips = [], [], [], []
    tot_iter = len(testloader)
    for val_step in range(tot_iter):
        _, val_inputs = next(testloader_iter)
        with torch.no_grad():
            val_inputs = [item.to(train_opt['device']) for item in val_inputs]
            if iteration >= fullbgid and len(val_inputs) == 4:
                xfull, xbg, xid, xmo = val_inputs
                val_inputs = [xfull, xfull, xfull, xmo] # Full BG ID
            output = model(val_inputs, is_training=False, writer=None)
        cur_ssim = output['ssim_metric'].clone().detach().cpu().numpy() if 'ssim_metric' in output.keys() else 0
        cur_msssim = output['msssim_metric'].clone().detach().cpu().numpy() if 'msssim_metric' in output.keys() else 0
        cur_rec = output['rec_loss'].clone().detach().cpu().numpy() if 'rec_loss' in output.keys() else 0
        cur_lpips = output['lpips_loss'].clone().detach().cpu().numpy() if 'lpips_loss' in output.keys() else 0
        ssim_metric.append(cur_ssim)
        msssim_metric.append(cur_msssim)
        rec_loss.append(cur_rec)
        lpips.append(cur_lpips)

        # print(f"SSIM: {cur_ssim}    MSE:  {cur_rec}    LPIPS: {cur_lpips}") # TODO!

        if val_step % 50 == 0 and dist.get_rank() == 0:
            rec_error = np.mean(rec_loss)
            ssim_simi = np.mean(ssim_metric)
            msssim_simi = np.mean(msssim_metric)
            lpips_simi = np.mean(lpips)
            Logger.info(f"{val_step}/{tot_iter}  rec:{rec_error} ssim:{ssim_simi} msssim:{msssim_simi} lpips:{lpips_simi}")
        if val_step == 0:
            x = get_visualize_img(val_inputs[0])
            x_rec = get_visualize_img(output['x_rec'])

    rec_error = np.mean(rec_loss)
    lpips_simi = np.mean(lpips)
    ssim_simi = np.mean(ssim_metric)
    msssim_simi = np.mean(msssim_metric)

    return rec_error, lpips_simi, ssim_simi, msssim_simi, time.time()-val_time, x, x_rec

def print_unused_params(model):
    for name, param in model.named_parameters():
        if param.grad is None or torch.all(param.grad==0):
            print(name)

if __name__ == '__main__':
    assert torch.cuda.is_available()

    parser = argparse.ArgumentParser(description="VQVAES")
    parser.add_argument('--opt', default=None, type=str, help="config file path")
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    opt = get_opt_from_yaml(args.opt)
    WANDB_OPEN = opt['train']['WANDB_OPEN']

    train_opt = opt['train']
    train_opt['local_rank'] = args.local_rank
    train_opt['device'] = torch.device('cuda', train_opt['local_rank'])

    print("Start to init torch distribution...")
    dist.init_process_group(backend='nccl', init_method='env://')
    print("Finish initializing torch distribution...")

    # get current experiment path
    setup_seed(10)
    if dist.get_rank() == 0:
        if opt['model']['checkpoint_path'] is not None and \
                train_opt['exp_name'] in os.path.basename(os.path.dirname(opt['model']['checkpoint_path'])):
            train_opt['exp_name'] = os.path.basename(os.path.dirname(opt['model']['checkpoint_path']))
            train_opt['save_path'] = os.path.abspath(os.path.dirname(opt['model']['checkpoint_path']))
            assert os.path.exists(train_opt['save_path'])
        else:
            if train_opt['exp_name'] is None:
                train_opt['exp_name'] = f"{opt['model']['name']}_{opt['dataset']['name']}"
            train_opt['exp_name'] = train_opt['exp_name'] + time.strftime("_%Y-%m-%d-%H-%M-%S", time.localtime())
            train_opt['save_path'] = os.path.join(train_opt['save_path'], train_opt['exp_name'])
            os.mkdir(train_opt['save_path'])
            os.mkdir(os.path.join(train_opt['save_path'], 'gt_img'))
            os.mkdir(os.path.join(train_opt['save_path'], 'rec_img'))

        # get experment info dir
        writer = SummaryWriter(os.path.join(train_opt['save_path'], 'log'))

        Logger = get_logger(os.path.join(train_opt['save_path'],
                                         'logging{}.log'.format(time.strftime("_%Y-%m-%d-%H-%M-%S", time.localtime()))),
                            train_opt['exp_name'], True)
        Logger.info(f"Loaded OPT file: {args.opt}")
        Logger.info('Experiment information is saved in %s' % train_opt['save_path'])
        Logger.info("Parameters' setting: ")
        print_opts(opt, Logger)

        if WANDB_OPEN:
            import wandb
            if opt['model']['checkpoint_path'] is not None and \
                    train_opt['exp_name'] == os.path.basename(os.path.dirname(opt['model']['checkpoint_path'])):
                project_name = train_opt['exp_name'] + time.strftime("_FOLLOW_%m%d%H%M%S", time.localtime())
            else:
                project_name = train_opt['exp_name']
            wandb.init(project="MoCoVQVAE", notes=str(opt),
                       group=opt['model']['name'].lower(),
                       tags=[opt['model']['name'].lower(),
                             opt['dataset']['name'].lower()],
                       entity="mzsun", id=project_name, resume='allow')
            wandb.config = {**opt}
    else:
        writer = None
        Logger = get_logger(logging_file=None, name=None, isopen=False)

    # load train data and test data
    trainloader, testloader, trainsampler, testsampler = get_data_loaders(opt)
    # opt['dataset']['name'] += '_EXT'
    # opt['dataset']['ret_mode'] = 'list'
    # trainloader2, testloader2, trainsampler2, testsampler2 = get_data_loaders(opt)

    # Initialize num_training updates befor get_model, caz some model needs the value.
    if train_opt['num_training_updates'] is None:
        train_opt['num_training_updates'] = int(train_opt['num_epochs'] * len(trainloader))
    elif train_opt['num_training_updates'] > int(train_opt['num_epochs'] * len(trainloader)):
        train_opt['num_training_updates'] = int(train_opt['num_epochs'] * len(trainloader))
    if train_opt['num_warmup_steps'] is None:
        train_opt['num_warmup_steps'] = int(0.1 * train_opt['num_training_updates'])
    Logger.info(f"epochs:{train_opt['num_epochs']}, " +
                f"tot_iteration: {train_opt['num_training_updates']}, " +
                f"warm_up:{train_opt['num_warmup_steps']}")

    # initialize model, optimizer and writer
    model, start_step = get_model(opt)
    model = model.to(train_opt['device'])
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[train_opt['local_rank']],
                                                output_device=train_opt['local_rank'],
                                                find_unused_parameters=train_opt['find_unused_parameters'])

    # optimizer & lr_scheduler
    Logger.info(f"Start to initialize optimizer and lrscheduler...")
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': train_opt['learning_rate']}],
                                 lr=train_opt['learning_rate'])
    lambda1 = lambda step: (step / train_opt['num_warmup_steps']) if step < train_opt['num_warmup_steps'] else 0.5 * (
            math.cos((step - train_opt['num_warmup_steps']) / (train_opt['num_training_updates'] - train_opt['num_warmup_steps']) * math.pi) + 1)
    if train_opt['LRsche'] is True:
        Logger.info(f"Use Learning rate scheduler!")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, last_epoch=start_step)
    else:
        scheduler = None

    # start iteration
    val_loss = -1
    stime = time.time()
    iteration = start_step
    fullbgid = opt['train'].get('fullbgid_start', opt['train']['disc_start_step'])
    Logger.info(f"Start to training with fullbgid_start: {fullbgid}...")

    # validation(model, testloader, train_opt, iteration)
    # validation(model, testloader2, train_opt, iteration)
    # st()

    for epoch in range(train_opt['num_epochs']):
        trainsampler.set_epoch(epoch)
        # if testsampler is not None:
        #     testsampler.set_epoch(epoch)
        for _, inputs in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()
            inputs = [item.to(train_opt['device']) for item in inputs]
            if iteration >= fullbgid:
                if len(inputs) == 4:
                    x, xbg, xid, xmo = inputs
                    inputs = [x, x, x, xmo] # Full BG ID
                else:
                    x, xco, xmo = inputs
                    inputs = [x, x, xmo]
            output = model(inputs, is_training=True,
                           writer=writer, optimizer=optimizer,
                           iteration=iteration, wandb_open=WANDB_OPEN)
            loss = output['loss']
            loss.backward()
            if opt['model']['name'] in ['MoCoVQVAE_wID', 'MoCoVQVAE_wCD']:
                if output['optimizer_idx'] == 0:
                    # 训练生成器时，不更新判别器的梯度
                    model.module._discriminator.zero_grad()
                if iteration >= opt['train']['disc_start_step'] and \
                        iteration < opt['train']['disc_start_step'] + 3 and dist.get_rank() == 0:
                    print(f"--------------------Find ununsed params for step {iteration}---------------")
                    print_unused_params(model.module)
            elif opt['model']['name'] in ['MoCoVQVAE_wICD']:
                if output['optimizer_idx'] == 0:
                    # 训练生成器时，不更新判别器的梯度
                    model.module._img_discriminator.zero_grad()
                    model.module._cat_discriminator.zero_grad()
                if iteration >= opt['train']['disc_start_step'] and \
                        iteration < opt['train']['disc_start_step'] + 3 and dist.get_rank() == 0:
                    print(f"--------------------Find ununsed params for step {iteration}---------------")
                    print_unused_params(model.module)
            else:
                if iteration < start_step + 3 and dist.get_rank() == 0:
                    print(f"--------------------Find ununsed params for step {iteration}---------------")
                    print_unused_params(model.module)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if (iteration + 1) % train_opt['num_validation_steps'] == 0 and testloader is not None and dist.get_rank()==0:
                Logger.info(f"====================validation start========================")
                Logger.info(f"Start to Validation for iter {iteration + 1}..")
                rec, p_loss, ssim, msssim, totime, x, x_rec = validation(model, testloader, train_opt, iteration)
                Logger.info(f"Validation finished after {totime}s.")
                Logger.info(f"rec_error:{rec} perceptual_loss:{p_loss}")
                Logger.info(f"SSIM_metric:{ssim} MSSSIM_metric:{msssim}")
                x.save(os.path.join(train_opt['save_path'], 'gt_img', str(iteration)+'.jpg'))
                x_rec.save(os.path.join(train_opt['save_path'], 'rec_img', str(iteration)+'.jpg'))
                if WANDB_OPEN:
                    show_x = wandb.Image(x, caption=f"valid x of {iteration}")
                    show_xrec = wandb.Image(x_rec, caption=f"valid x_rec of {iteration}")
                    wandb.log({'valid_x': show_x, 'valid_xrec': show_xrec}, step=iteration)
                    wandb.log({'valid_ssim': ssim, 'valid_msssim':msssim}, step=iteration)
                    wandb.log({"valid_p_loss": p_loss}, step=iteration)
                    wandb.log({"valid_rec_loss": rec}, step=iteration)
                Logger.info(f"====================validation  end========================")

            if (iteration + 1) % 50 == 0 and dist.get_rank() == 0:
                left_time = (time.time() - stime) / (iteration - start_step) * (
                            train_opt['num_training_updates'] - iteration)
                Logger.info("{}/{}/{} lr:{:.5f} loss:{:.2f} val:{:.2f} left_time:{:.2f}h".format(epoch, iteration,
                                                                                                 train_opt['num_training_updates'],
                                                                                                 optimizer.state_dict()[
                                                                                                     'param_groups'][0][
                                                                                                     'lr'],
                                                                                                 loss, val_loss,
                                                                                                 left_time / 3600))

            if torch.distributed.get_rank() == 0 and (iteration + 1) % train_opt['save_ckpt_per_iter'] == 0:
                torch.save({'state': model.state_dict(), 'steps': iteration + 1},
                           os.path.join(train_opt['save_path'], "{}_iter{}.pth".format(opt['model']['name'], iteration + 1)))

            iteration += 1
            if iteration >= train_opt['num_training_updates']:
                break

    if torch.distributed.get_rank() == 0:
        Logger.info("Saving final model")
        torch.save({'state': model.state_dict(), 'steps': iteration + 1},
                   os.path.join(train_opt['save_path'], f"{opt['model']['name']}_final.pth"))
        Logger.info("Training has finished!")

    Logger.info(f"====================validation start========================")
    Logger.info(f"Start to Validation for iter {iteration + 1}..")
    rec, p_loss, ssim, msssim, totime, x, x_rec = validation(model, testloader, train_opt, iteration)
    Logger.info(f"Validation finished after {totime}s.")
    Logger.info(f"rec_error:{rec} perceptual_loss:{p_loss}")
    Logger.info(f"SSIM_metric:{ssim} MSSSIM_metric:{msssim}")
    x.save(os.path.join(train_opt['save_path'], 'gt_img', str(iteration) + '.jpg'))
    x_rec.save(os.path.join(train_opt['save_path'], 'rec_img', str(iteration) + '.jpg'))
    if WANDB_OPEN:
        show_x = wandb.Image(x, caption=f"valid x of {iteration}")
        show_xrec = wandb.Image(x_rec, caption=f"valid x_rec of {iteration}")
        wandb.log({'valid_x': show_x, 'valid_xrec': show_xrec}, step=iteration)
        wandb.log({'valid_ssim': ssim, 'valid_msssim': msssim}, step=iteration)
        wandb.log({"valid_p_loss": p_loss}, step=iteration)
        wandb.log({"valid_rec_loss": rec}, step=iteration)
    Logger.info(f"====================validation  end========================")

if __name__ == '__main__':
    main()