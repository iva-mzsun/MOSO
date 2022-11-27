import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.distributed as dist
from src.utils import get_logger
from .SingleMoDataset import SingleMoDataset
from .UCFClassDataset import UCFClassDataset
from .CustomVPDataset import CustomVPDataset
from .CustomVPDataset_FixPreMO import CustomVPDataset_FixPreMo
from .CustomUnconditionalDataset import CustomUnconditionalDataset

def get_dataloader(opt):
    trainset, validset = get_dataset(opt.dataset)

    train_sampler = Data.distributed.DistributedSampler(trainset)
    trainloader = Data.DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=opt.train.batch_size,
        pin_memory=opt.dataset.pin_memory,
        shuffle=opt.dataset.shuffle,
        num_workers=opt.dataset.num_worker,
        drop_last=True
    )

    valid_sampler = Data.distributed.DistributedSampler(validset)
    validloader = Data.DataLoader(
        validset,
        sampler=valid_sampler,
        batch_size=opt.train.batch_size,
        pin_memory=opt.dataset.pin_memory,
        shuffle=opt.dataset.shuffle,
        num_workers=opt.dataset.num_worker,
        drop_last=True,
    )

    return trainloader, train_sampler, validloader, valid_sampler


def get_dataset(dataset_opt):
    if dataset_opt.cname == 'CustomUnconditionalDataset':
        trainset = CustomUnconditionalDataset(tokens_dir=dataset_opt.train.tokens_dir)
        validset = CustomUnconditionalDataset(tokens_dir=dataset_opt.valid.tokens_dir)
    elif dataset_opt.cname == 'CustomVPDataset':
        trainset = CustomVPDataset(tokens_1th_dir=dataset_opt.train.tokens_1th_dir,
                                   tokens_16f_dir=dataset_opt.train.tokens_16f_dir)
        validset = CustomVPDataset(tokens_1th_dir=dataset_opt.valid.tokens_1th_dir,
                                   tokens_16f_dir=dataset_opt.valid.tokens_16f_dir)
    elif dataset_opt.cname == 'CustomVPDataset_FixPreMo':
        trainset = CustomVPDataset_FixPreMo(tokens_pre_dir=dataset_opt.train.tokens_pre_dir,
                                            tokens_tar_dir=dataset_opt.train.tokens_tar_dir,
                                            pre_frame_num=dataset_opt.pre_frame_num)
        validset = CustomVPDataset_FixPreMo(tokens_pre_dir=dataset_opt.valid.tokens_pre_dir,
                                            tokens_tar_dir=dataset_opt.valid.tokens_tar_dir,
                                            pre_frame_num=dataset_opt.pre_frame_num)
    elif dataset_opt.cname == 'SingleMoDataset':
        trainset = SingleMoDataset(tokens_dir=dataset_opt.train.tokens_dir)
        validset = SingleMoDataset(tokens_dir=dataset_opt.valid.tokens_dir)
    elif dataset_opt.cname == 'UCFClassDataset':
        trainset = UCFClassDataset(class2id=dataset_opt.train.class2id,
                                   tokens_dir=dataset_opt.train.tokens_dir)
        validset = UCFClassDataset(class2id=dataset_opt.valid.class2id,
                                   tokens_dir=dataset_opt.valid.tokens_dir)
    else:
        raise NotImplementedError(f"Dataset: {dataset_opt.cname}")

    log = get_logger()
    log.info(f"Finish Loading dataset class: {dataset_opt.cname} with train data: {len(trainset)} and valid data: {len(validset)}")

    return trainset, validset
