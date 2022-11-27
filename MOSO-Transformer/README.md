##MoCoVQVAE
1、训练：
~~~
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=10001 \
moco_vqvae/train_dist.py --opt moco_vqvae/config/MoCoVQVAEwCD/UCF101/MoCoVQVAEwCD_im256_8frames.yaml
~~~

2、测试重构IS/FID：
~~~
CUDA_VISIBLE_DEVICE=1 python moco_vqvae/ISFID_Reconstruction.py \
--opt moco_vqvae/config/MoCoVQVAEwCD/UCF101/MoCoVQVAEwCD_im256_8frames.yaml \
--ckpt /home/zhongguokexueyuanzidonghuayanjiusuo/mzsun/codes/MoCoVQVAE/experiments/MoCoVQVAEwCD_UCF_im256_8frames_2022-05-15-20-47-40/MoCoVQVAE_wCD_iter250000.pth
~~~

3、统计模型的参数量/计算量：
~~~
CUDA_VISIBLE_DEVICES=1 python moco_vqvae/test_model.py \
--opt moco_vqvae/config/MoCoVQVAEwCD/UCF101/MoCoVQVAEwCD_im256_8frames.yaml
~~~