# KITTI 256x

## 生成可视化样例
deepspeed --include=localhost:4 --master_port=10004 \
video_predict.py --seed 10 --timesteps 64 --temperature 0 \
--split train --iterative 2 --vp_base_frames 5 --batchsize 2 --recode \
--opt config/StackTRM2/KITTI/base_256x_5to15_noisy_l14+8.yaml \
--ckpt experiments/KITTI_256x_5to15_noise_l14+8_20221101-084831/ckpt/000190000 \
--items_json /raid/mzsun/codes/MOSO/process_and_eval/KITTI256x/trainitems_for_vp_n2.json \
--deepspeed --deepspeed_config config/deepspeed/deepspeed_base_fp16.json

