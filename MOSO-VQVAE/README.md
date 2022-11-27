# MoCoVQVAE

### pipeline
以FaceForensics数据集为例，视频存储格式：
```
dataset_path:
    train: 
        video0:
            frame0.png
            frame2.png
            ...
        video1:
            frame0.png
            ...
        ...
    valid:
        video0:
            frame0.png
            frame2.png
            ...
        video1:
            frame0.png
            ...
        ...
```

1、配置文件MoCoVQVAEwCD_im128_woPT_dbBSLR.yaml，训练MoCoVQVAE。运行命令：
```
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --node_rank=1 --master_port=10000 \
train_dist.py --opt config/MoCoVQVAEwCD/FaceForensics/MoCoVQVAEwCD_im128_woPT_dbBSLR.yaml
```

2、基于训练好的模型和加载的train/valid dataloader，提取train和valid的视频Token
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --master_port=10000 \
extract_tokens.py --opt config/MoCoVQVAEwCD/FaceForensics/MoCoVQVAEwCD_im128_woPT_dbBSLR.yaml \
--save_dir /home/zhongguokexueyuanzidonghuayanjiusuo/mzsun/codes/MoCo/MoCo_VG/datasets/FaceForensics/im128_token \
--ckpt experiments/MoCoVQVAEwCD_Face_im128_woPT_dbBS\&LR_2022-05-31-15-39-07/MoCoVQVAE_wCD_iter40000.pth
```