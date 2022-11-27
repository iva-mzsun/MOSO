# BAIR
deepspeed --hostfile=config/deepspeed/hostfile \
--include="gpu12:4,5,6,7" --master_port=14567 \
train.py --opt config/StackTRM2/BAIR/base_64rs_16f_fps32_stack_wd.yaml \
--deepspeed --deepspeed_config config/deepspeed/deepspeed_base_fp16_bs8_lr5e-5.json

deepspeed --hostfile=config/deepspeed/hostfile --master_port=9000 --include="cli02:0" \
video_predict.py --seed 0 --num_samples 100 --timesteps 16 --temperature 3 \
--split test --items_json datasets/BAIR/vp_test_100items.json \
--opt config/StackTRM2/BAIR/base_64rs_16f_fps32_stack_wd.yaml \
--ckpt experiments/BAIR_64rs_16f_fps32_stacktrm2_wd_20220715-000436/ckpt/000055000 \
--deepspeed --deepspeed_config config/deepspeed/deepspeed_base_fp16.json


# RoboNet
deepspeed --hostfile=config/deepspeed/hostfile \
--include="gpu13:0,1,2,3" --master_port=10123 \
train.py --opt config/StackTRM2/RoboNet/base_64rs_16f_fps32_stack_wd.yaml \
--deepspeed --deepspeed_config config/deepspeed/deepspeed_base_fp16_bs8_lr5e-5.json

deepspeed --hostfile=config/deepspeed/hostfile --master_port=9000 --include="cli02:0" \
video_predict.py --seed 0 --num_samples 100 --timesteps 16 --temperature 3 \
--split test --items_json datasets/RoboNet/tokens/vq_test_items100.json \
--opt config/StackTRM2/RoboNet/base_64rs_16f_fps32_stack_wd.yaml \
--ckpt experiments/RoboNet_64rs_16f_fps32_stacktrm2_wd_20220715-121745/ckpt/000055000 \
--deepspeed --deepspeed_config config/deepspeed/deepspeed_base_fp16.json



