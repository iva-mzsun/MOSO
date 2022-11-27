# Face 256rs 16f id4
deepspeed --hostfile=config/deepspeed/hostfile --master_port=10000 --include="gpu13:0" \
generate.py --seed 0 --num_samples 100 \
--timesteps 16 --batchsize 2 --mode sample --temperature 4.5 \
--opt config/StackTRM2/FaceForensics/base_256rs_16f_fp16_splittrain.yaml \
--ckpt experiments/Face_256rs_16f_splittrain_20220714-230707/ckpt/000200000 \
--deepspeed --deepspeed_config config/deepspeed/deepspeed_base_fp16.json

python src/scripts/calc_metrics_for_dataset.py \
--real_data_path vqvae_experiments/MoCoVQVAEwCD_Face_im256_16frames_id4_2022-06-09-23-57-05/MoCoVQVAE_wCD_iter250000_gtframes \
--fake_data_path git_experiments/Face_256rs_16f_splittrain_20220714-230707/sample_videos/CKPT200000-T16-TYPEdefault-TEMP4.5-2STAGE \
--mirror 1 --gpus 1 --resolution 256 --metrics fvd2048_16f --verbose 1 --use_cache 0
