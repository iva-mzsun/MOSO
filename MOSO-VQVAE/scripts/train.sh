ssh gpu13
conda activate edvr
cd mzsun/codes/MoCoVQVAE

# General
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --nnodes=1 --master_port=10012 \
train_dist.py --opt config/mocovqvae_wcd_sCB/General/MoCoVQVAEwCD_im256_16frames_id4.yaml

# KTH
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10012 \
train_dist.py --opt config/mocovqvae_wcd_sCB/KTH/MoCoVQVAEwCD_im64_20frames_th10.yaml

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10012 \
train_dist.py --opt config/mocovqvae_wcd_sCB/KTH/Abalate/MoCoVQVAEwCD_im64_20frames.yaml

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10023 \
train_dist.py --opt config/mocovqvae_wcd_sCB/KTH/Abalate/MoCoVQVAEwCD_im64_20frames_mo.yaml

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10045 \
train_dist.py --opt config/mocovqvae_wcd_sCB/KTH/Abalate/MoCoVQVAEwCD_im64_20frames_como.yaml


# KITTI
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10023 \
train_dist.py --opt config/mocovqvae_wcd/KITTI/MoCoVQVAEwCD_im64_f2_20frames.yaml

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10001 \
train_dist.py --opt config/mocovqvae_wcd_sCB/KITTI/MoCoVQVAEwCD_im64_f2_20frames.yaml

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10045 \
train_dist.py --opt config/mocovqvae_wcd_sCB/KITTI/MoCoVQVAEwCD_im64_f2_10frames.yaml

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10045 \
train_dist.py --opt config/mocovqvae_wcd_sCB/KITTI/MoCoVQVAEwCD_im64_f2_10frames_cb1W_como.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --nnodes=1 --master_port=10001 \
train_dist.py --opt config/mocovqvae_wcd_sCB/KITTI/MoCoVQVAEwCD_im256_f2_20frame_wDISC.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=4 --nnodes=1 --master_port=10002 \
train_dist.py --opt config/mocovqvae_wcd_sCB/KITTI/MoCoVQVAEwCD_im256_f2_15frame_wDISC.yaml


# RoboNet
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10067 \
train_dist.py --opt config/mocovqvae_wcd_sCB/RoboNet/MoCoVQVAEwCD_im64_16frames_co+mo.yaml

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10067 \
train_dist.py --opt config/mocovqvae_wcd_sCB/RoboNet/MoCoVQVAEwCD_im64_16frames.yaml

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10067 \
train_dist.py --opt config/mocovqvae_wcd_sCB/RoboNet/MoCoVQVAEwCD_im256_12frames.yaml

# BAIR
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10045 \
train_dist.py --opt config/mocovqvae_wcd_sCB/BAIR/MoCoVQVAEwCD_im64_16frames.yaml

# Face
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10067 \
train_dist.py --opt config/mocovqvae_wcd_sCB/FaceForensics/MoCoVQVAEwCD_im256_16frames_id4.yaml

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10023 \
train_dist.py --opt config/mocovqvae_wcd_sCB/FaceForensics/MoCoVQVAEwCD_im256_16frames_id4_2.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=4 --nnodes=1 --master_port=14567 \
train_dist.py --opt config/mocovqvae_wcd/FaceForensics/MoCoVQVAEwCD_im256_16frames_id4_woFixPre.yaml

# SkyTimelapse
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10045 \
train_dist.py --opt config/mocovqvae_wcd_sCB/SkyTimelapse/MoCoVQVAEwCD_im256_16frames_id4.yaml

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10045 \
train_dist.py --opt config/mocovqvae_wcd_sCB/SkyTimelapse/MoCoVQVAEwCD_im128_16frames_fps32.yaml

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10067 \
train_dist.py --opt config/mocovqvae_wcd_sCB/SkyTimelapse/MoCoVQVAEwCD_im128_16frames_fps4+8+16+32.yaml

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10067 \
train_dist.py --opt config/mocovqvae_wcd_sCB/SkyTimelapse/MoCoVQVAEwCDwTA_im128_16frames_fps4+8+16+32.yaml

# UCF
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=4 --nnodes=1 --master_port=10001 \
train_dist.py --opt config/mocovqvae_wcd_sCB/UCF101/MoCoVQVAEwCD_im256_6frames_id4_woDISC.yaml

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10001 \
train_dist.py --opt config/mocovqvae_wcd_sCB/UCF101/MoCoVQVAEwCD_im256_16frames_id4.yaml

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10023 \
train_dist.py --opt config/mocovqvae_wcd_sCB/UCF101/MoCoVQVAEwCD_im256_16frames_id4_como.yaml

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10045 \
train_dist.py --opt config/mocovqvae_wcd_sCB/UCF101/MoCoVQVAEwCD_im256_16frames_id4_mo.yaml

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10067 \
train_dist.py --opt config/mocovqvae_wcd_sCB/UCF101/MoCoVQVAEwCD_im256_16frames_id4_como2.yaml


## Webvid
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --master_port=10007 \
train_dist.py --opt config/MoCoVQVAEwCD/UCF101/MoCoVQVAEwCD_DS4_FullBGID.yaml
