# test refine file
#deepspeed --hostfile=config/deepspeed/hostfile --master_port=14321 --include="cli02:0" \
#refine.py --seed 0 --timesteps 8 --mode test --not_show_process --batchsize 1 \
#--opt config/base_deepspeed/FaceForensics/base_256rs_16f_fp16_sepmo.yaml \
#--ckpt experiments/Face_256rs_16f_fp16_sepmo_20220612-142149/ckpt/000300000 \
#--deepspeed --deepspeed_config config/deepspeed/deepspeed_base_fp16.json \
#--src_tokens_file experiments/Face_256rs_16f_fp16_sepmo_20220612-142149/sample_tokens/trainsteps300000-timesteps8-temp0.1/sample11-batch0-seed0.npy \
#--cut_prob 0.99

# test refine dir
deepspeed --hostfile=config/deepspeed/hostfile --include="cli02:0" \
refine.py --seed 0 --timesteps 2 --mode sample --batchsize 1 \
--opt config/StackTRM/FaceForensics/base_256rs_16f_fp16_stack.yaml \
--ckpt experiments/Face_256rs_16f_fp16_stack_20220615-130641/ckpt \
--deepspeed --deepspeed_config config/deepspeed/deepspeed_base_fp16.json \
--src_tokens_dir Face_256rs_16f_fp16_stack_20220615-130641/sample_tokens/CKPT220000-T16-TYPEwoConfidence-TEMP0.6-2STAGE

deepspeed --hostfile=config/deepspeed/hostfile --include="cli02:0" \
refine.py --seed 0 --timesteps 4 --mode sample --batchsize 1 \
--opt config/StackTRM/FaceForensics/base_256rs_16f_fp16_stack.yaml \
--ckpt experiments/Face_256rs_16f_fp16_stack_20220615-130641/ckpt \
--deepspeed --deepspeed_config config/deepspeed/deepspeed_base_fp16.json \
--src_tokens_dir Face_256rs_16f_fp16_stack_20220615-130641/sample_tokens/CKPT220000-T16-TYPEwoConfidence-TEMP0.6-2STAGE

deepspeed --hostfile=config/deepspeed/hostfile --include="cli02:0" \
refine.py --seed 0 --timesteps 8 --mode sample --batchsize 1 \
--opt config/StackTRM/FaceForensics/base_256rs_16f_fp16_stack.yaml \
--ckpt experiments/Face_256rs_16f_fp16_stack_20220615-130641/ckpt \
--deepspeed --deepspeed_config config/deepspeed/deepspeed_base_fp16.json \
--src_tokens_dir Face_256rs_16f_fp16_stack_20220615-130641/sample_tokens/CKPT220000-T16-TYPEwoConfidence-TEMP0.6-2STAGE

deepspeed --hostfile=config/deepspeed/hostfile --include="cli02:0" \
refine.py --seed 0 --timesteps 16 --mode sample --batchsize 1 \
--opt config/StackTRM/FaceForensics/base_256rs_16f_fp16_stack.yaml \
--ckpt experiments/Face_256rs_16f_fp16_stack_20220615-130641/ckpt \
--deepspeed --deepspeed_config config/deepspeed/deepspeed_base_fp16.json \
--src_tokens_dir Face_256rs_16f_fp16_stack_20220615-130641/sample_tokens/CKPT220000-T16-TYPEwoConfidence-TEMP0.6-2STAGE

deepspeed --hostfile=config/deepspeed/hostfile --master_port=10001 --include="gpu13:1" \
refine.py --seed 0 --timesteps 8 --mode sample --batchsize 1 \
--opt config/base_deepspeed/FaceForensics/base_256rs_16f_fp16_sepmo.yaml \
--ckpt experiments/Face_256rs_16f_fp16_sepmo_20220612-142149/ckpt/000300000 \
--deepspeed --deepspeed_config config/deepspeed/deepspeed_base_fp16.json \
--src_tokens_dir experiments/Face_256rs_16f_fp16_sepmo_20220612-142149/sample_tokens/trainsteps300000-timesteps8-temp0.9 \
--cut_prob 0.99
