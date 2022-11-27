

####### BAIR - 64rs - 16f ########
# IS/FID
CUDA_VISIBLE_DEVICE=4 python ISFID_Reconstruction.py \
--opt config/mocovqvae_wcd_sCB/BAIR/MoCoVQVAEwCD_im64_16frames.yaml \
--ckpt experiments/MoCoVQVAEwCDsCB_BAIR_im64_16frames_2022-07-06-10-29-50/MoCoVQVAE_wCD_shareCB_iter250000.pth --if_save True
# FVD
python src/scripts/calc_metrics_for_dataset.py \
--real_data_path vqvae_experiments/MoCoVQVAEwCDsCB_BAIR_im64_16frames_2022-07-06-10-29-50/MoCoVQVAE_wCD_shareCB_iter250000_gtframes \
--fake_data_path vqvae_experiments/MoCoVQVAEwCDsCB_BAIR_im64_16frames_2022-07-06-10-29-50/MoCoVQVAE_wCD_shareCB_iter250000_recframes \
--mirror 1 --gpus 1 --resolution 64 --metrics fvd2048_16f --verbose 1 --use_cache 0
# Extract Tokens
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=10004 \
extract_tokens.py --opt config/mocovqvae_wcd_sCB/BAIR/MoCoVQVAEwCD_im64_16frames.yaml \
--save_dir datasets/BAIR/tokens/im64_bg2_id2_mo3 \
--ckpt experiments/MoCoVQVAEwCDsCB_BAIR_im64_16frames_2022-07-06-10-29-50/MoCoVQVAE_wCD_shareCB_iter250000.pth

####### Sky - 256rs - 16f - id4 ########
# IS/FID
CUDA_VISIBLE_DEVICE=7 python ISFID_Reconstruction.py \
--opt config/mocovqvae_wcd_sCB/SkyTimelapse/MoCoVQVAEwCD_im256_16frames_id4.yaml \
--ckpt experiments/MoCoVQVAEwCDsCB_Sky_im256_16frames_id4_2022-06-25-14-41-43/MoCoVQVAE_wCD_shareCB_iter250000.pth
# FVD
python src/scripts/calc_metrics_for_dataset.py \
--real_data_path vqvae_experiments/MoCoVQVAEwCDsCB_Sky_im256_16frames_id4_2022-06-25-14-41-43/MoCoVQVAE_wCD_shareCB_iter250000_gtframes \
--fake_data_path vqvae_experiments/MoCoVQVAEwCDsCB_Sky_im256_16frames_id4_2022-06-25-14-41-43/MoCoVQVAE_wCD_shareCB_iter250000_recframes \
--mirror 1 --gpus 1 --resolution 256 --metrics fvd2048_16f,fid50k_full --verbose 1 --use_cache 0
# Extract Tokens
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=10005 \
extract_tokens.py --opt config/mocovqvae_wcd_sCB/SkyTimelapse/MoCoVQVAEwCD_im256_16frames_id4.yaml \
--save_dir datasets/SkyTimelapse/tokens/im256_bg3_id4_mo5 --cur_fps 4 \
--ckpt experiments/MoCoVQVAEwCDsCB_Sky_im256_16frames_id4_2022-06-25-14-41-43/MoCoVQVAE_wCD_shareCB_iter250000.pth

####### Sky - 128rs - 16f - fps4/8/16/32 - id4 ########
# Train
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10045 \
train_dist.py --opt config/mocovqvae_wcd_sCB/SkyTimelapse/MoCoVQVAEwCD_im128_16frames_fps4+8+16+32.yaml
# Extract Tokens
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=10005 \
extract_tokens.py \
--opt config/mocovqvae_wcd_sCB/SkyTimelapse/MoCoVQVAEwCD_im128_16frames_fps4+8+16+32.yaml \
--save_dir datasets/SkyTimelapse/tokens/img128_bg3_id3_mo4_fps4+8+16+32 \
--ckpt experiments/MoCoVQVAEwCDsCB_Sky_im128_16frames_fps4+8+16+32_2022-07-02-13-35-16/MoCoVQVAE_wCD_shareCB_iter250000.pth

####### Sky - 128rs - 16f - fps32 - id4 ########
# Train
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10045 \
train_dist.py --opt config/mocovqvae_wcd_sCB/SkyTimelapse/MoCoVQVAEwCD_im128_16frames_fps32.yaml
# IS/FID
CUDA_VISIBLE_DEVICE=4 python ISFID_Reconstruction.py \
--opt config/mocovqvae_wcd_sCB/SkyTimelapse/MoCoVQVAEwCD_im128_16frames_fps32.yaml \
--ckpt experiments/MoCoVQVAEwCDsCB_Sky_im128_16frames_fps32_2022-07-02-13-35-16/MoCoVQVAE_wCD_shareCB_iter250000.pth
# Extract Tokens
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=10005 \
extract_tokens.py  # --skip_valid \
--opt config/mocovqvae_wcd_sCB/SkyTimelapse/MoCoVQVAEwCD_im128_16frames_fps32.yaml \
--save_dir datasets/SkyTimelapse/tokens/im128_bg3_id3_mo4_fps32only \
--ckpt experiments/MoCoVQVAEwCDsCB_Sky_im128_16frames_fps32_2022-07-02-13-35-16/MoCoVQVAE_wCD_shareCB_iter250000.pth

####### UCF - 256rs - 16f - id4######
# Train
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --master_port=10067 \
train_dist.py --opt config/mocovqvae_wcd_sCB/UCF101/MoCoVQVAEwCD_im256_16frames_id4.yaml
# IS/FID
CUDA_VISIBLE_DEVICE=6 python ISFID_Reconstruction.py \
--opt config/mocovqvae_wcd_sCB/UCF101/MoCoVQVAEwCD_im256_16frames_id4.yaml \
--ckpt experiments/MoCoVQVAEwCDsCB_UCF_im256_16frames_id4_2022-06-29-13-08-50/MoCoVQVAE_wCD_shareCB_iter250000.pth --if_save True
# extract tokens
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=10007 \
extract_tokens.py --opt config/mocovqvae_wcd_sCB/UCF101/MoCoVQVAEwCD_im256_16frames_id4.yaml \
--save_dir datasets/UCF101/tokens/ \
--ckpt experiments/MoCoVQVAEwCDsCB_UCF_im256_16frames_id4_2022-06-29-13-08-50/MoCoVQVAE_wCD_shareCB_iter250000.pth

####### Face - 256rs - 16f - id4 ########
# IS/FID
CUDA_VISIBLE_DEVICE=2 python ISFID_Reconstruction.py \
--opt config/mocovqvae_wcd_sCB/FaceForensics/MoCoVQVAEwCD_im256_16frames_id4.yaml \
--ckpt experiments/MoCoVQVAEwCDsCB_Face_im256_16frames_id4_2022-06-23-10-22-57/MoCoVQVAE_wCD_shareCB_iter250000.pth --if_save True
# FVD
# cd /home/zhongguokexueyuanzidonghuayanjiusuo/mzsun/codes/metrics
python calcu_fvd.py --max_frames 16 \
--gt_path experiments/MoCoVQVAEwCD_Face_im256_16frames_id4_2022-06-12-23-44-09/MoCoVQVAE_wCD_iter250000_gtframes \
--gen_path experiments/MoCoVQVAEwCD_Face_im256_16frames_id4_2022-06-12-23-44-09/MoCoVQVAE_wCD_iter250000_recframes
# extract tokens
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=10001 \
extract_tokens.py --opt config/mocovqvae_wcd/FaceForensics/MoCoVQVAEwCD_im256_16frames_id4.yaml \
--save_dir datasets/FaceForensics/Face_tokens_im256_bg3_id4_mo5/ \
--ckpt experiments/MoCoVQVAEwCD_Face_im256_16frames_id4_2022-06-12-23-44-09/MoCoVQVAE_wCD_iter250000.pth
# visualize tokens
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=10001 \
visualize_tokens.py --opt config/mocovqvae_wcd/FaceForensics/MoCoVQVAEwCD_im256_16frames_id4.yaml \
--ckpt experiments/MoCoVQVAEwCD_Face_im256_16frames_id4_2022-06-12-23-44-09/MoCoVQVAE_wCD_iter250000.pth \
--tokens_dir datasets/FaceForensics/Face_tokens_im256_bg3_id4_mo5/valid

####### Face - 256rs - 16f ########
# IS/FID
CUDA_VISIBLE_DEVICE=0 python ISFID_Reconstruction.py \
--opt config/mocovqvae_wcd/FaceForensics/MoCoVQVAEwCD_im256_16frames.yaml \
--ckpt experiments/MoCoVQVAEwCD_Face_im256_16frames_2022-06-09-23-56-54/MoCoVQVAE_wCD_iter250000.pth --if_save True
# FVD
# cd /home/zhongguokexueyuanzidonghuayanjiusuo/mzsun/codes/metrics
python calcu_fvd.py --max_frames 16 \
--gt_path experiments/MoCoVQVAEwCD_Face_im256_16frames_2022-06-09-23-56-54/MoCoVQVAE_wCD_iter250000_gtframes \
--gen_path experiments/MoCoVQVAEwCD_Face_im256_16frames_2022-06-09-23-56-54/MoCoVQVAE_wCD_iter250000_recframes
# extract tokens
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=10001 \
extract_tokens.py --opt config/mocovqvae_wcd/FaceForensics/MoCoVQVAEwCD_im256_16frames.yaml \
--save_dir datasets/FaceForensics/Face_tokens_im256_bg3_id3_mo5/ \
--ckpt experiments/MoCoVQVAEwCD_Face_im256_16frames_2022-06-09-23-56-54/MoCoVQVAE_wCD_iter250000.pth
# visualize tokens
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=10001 \
visualize_tokens.py --opt config/mocovqvae_wcd/FaceForensics/MoCoVQVAEwCD_im256_16frames.yaml \
--ckpt experiments/MoCoVQVAEwCD_Face_im256_16frames_2022-06-09-23-56-54/MoCoVQVAE_wCD_iter250000.pth \
--tokens_dir datasets/FaceForensics/Face_tokens_im256_bg3_id3_mo5/valid

####### UCF - 256rs - 16f - id4######
# IS/FID
CUDA_VISIBLE_DEVICE=0 python ISFID_Reconstruction.py \
--opt config/mocovqvae_wcd_sCB/UCF101/MoCoVQVAEwCD_im128_8frames_shareCB.yaml \
--ckpt experiments/MoCoVQVAEwCDsCB_UCF_im128_8frames_cb8192_2022-06-16-16-08-02/MoCoVQVAE_wCD_shareCB_iter250000.pth
# FVD
# cd /home/zhongguokexueyuanzidonghuayanjiusuo/mzsun/codes/metrics
python calcu_fvd.py --max_frames 16 \
--gt_path experiments/MoCoVQVAEwCD_UCF_im256_16frames_id4_2022-06-12-23-44-09/MoCoVQVAE_wCD_iter250000_gtframes \
--gen_path experiments/MoCoVQVAEwCD_UCF_im256_16frames_id4_2022-06-12-23-44-09/MoCoVQVAE_wCD_iter250000_recframes
# extract tokens
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=10001 \
extract_tokens.py --opt config/mocovqvae_wcd/FaceForensics/MoCoVQVAEwCD_im256_16frames_id4.yaml \
--save_dir /home/zhongguokexueyuanzidonghuayanjiusuo/mzsun/codes/MoCo/datasets/FaceForensics/Face_tokens_im256_bg3_id4_mo5/ \
--ckpt experiments/MoCoVQVAEwCD_UCF_im256_16frames_id4_2022-06-12-23-44-09/MoCoVQVAE_wCD_iter250000.pth
# visualize tokens
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=10001 \
visualize_tokens.py --opt config/mocovqvae_wcd/FaceForensics/MoCoVQVAEwCD_im256_16frames_id4.yaml \
--ckpt experiments/MoCoVQVAEwCD_UCF_im256_16frames_id4_2022-06-06-16-45-15/MoCoVQVAE_wCD_iter250000.pth \
--tokens_dir /home/zhongguokexueyuanzidonghuayanjiusuo/xinxin.zhu/mzsun/codes/MoCo/datasets/UCF101/UCF101_tokens_im256_bg3_id4_mo5/valid


