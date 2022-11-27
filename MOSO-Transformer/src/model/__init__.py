from src.utils import get_logger
# from src.model.Base.MoCoTransformer_Base import MoCoTransformer_Base as Base
# from src.model.Base_deepspeed.MoCoTransformer_Base import MoCoTransformer_Base as Base_DS
# from src.model.Base_deepspeed_class.MoCoTransformer_Base import MoCoTransformer_Base as Base_DS_Class
from src.model.StackTRM.MoCoTransformer import MoCoTransformer as stacktrm
from src.model.StackTRM2.MoCoTransformer import MoCoTransformer as stacktrm2
from src.model.StackTRM2.splittrain_MoCoTransformer import MoCoTransformer_st as stacktrm2_st
from src.model.StackTRM2_shareCB.MoCoTransformer import MoCoTransformer as stacktrm2_shareCB
from src.model.StackTRM2_sCB_VP.MoCoTransformer import MoCoTransformer as stacktrm2_shareCB_VP
from src.model.StackTRM2_sCB_VP.MoCoTransformer_weightDIFF import MoCoTransformer as stacktrm2_shareCB_VP_wd
from src.model.StackTRM2_sCB_VP_noise.MoCoTransformer_weightDIFF import MoCoTransformer as stacktrm2_shareCB_VP_wd_noise
from src.model.StackTRM2sCB_VP_fixpremo.MoCoTransformer_weightDIFF import MoCoTransformer as stacktrm2_shareCB_VP_wd_fixpremo
from src.model.StackTRM2sCB_VP_fixpremo_noise.MoCoTransformer_weightDIFF import MoCoTransformer as stacktrm2_shareCB_VP_wd_fixpremo_noise
from src.model.StackTRM2sCB_VP_fixpremo_mo.MoCoTransformer_weightDIFF import MoCoTransformer as stacktrm2_shareCB_VP_wd_fixpremo_mo
from src.model.StackTRM2sCB_VP_fixpremo_moco.MoCoTransformer_weightDIFF import MoCoTransformer as stacktrm2_shareCB_VP_wd_fixpremo_moco
from src.model.StackTRM_VQemb.MoCoTransformer import MoCoTransformer as stacktrm_vqemb

def get_model(opt, load_vqvae):
    log = get_logger()

    if opt.model.name.lower() in ['base']:
        log.info(f"Start to load model: {opt.model.name}...")
        model = Base(
            train_opt=opt.train,
            model_opt=opt.model,
            vqvae_opt=opt.vqvae,
            tokenizer_opt=opt.tokenizer,
            generation_opt=opt.generation,
            load_vqvae=load_vqvae
        )
    elif opt.model.name.lower() in ['stacktrm']:
        log.info(f"Start to load model: {opt.model.name}...")
        model = stacktrm(
            train_opt=opt.train,
            model_opt=opt.model,
            vqvae_opt=opt.vqvae,
            tokenizer_opt=opt.tokenizer,
            generation_opt=opt.generation,
            load_vqvae=load_vqvae
        )
    elif opt.model.name.lower() in ['stacktrm2']:
        log.info(f"Start to load model: {opt.model.name}...")
        model = stacktrm2(
            train_opt=opt.train,
            model_opt=opt.model,
            vqvae_opt=opt.vqvae,
            tokenizer_opt=opt.tokenizer,
            generation_opt=opt.generation,
            load_vqvae=load_vqvae
        )
    elif opt.model.name.lower() in ['stacktrm2_st']:
        log.info(f"Start to load model: {opt.model.name}...")
        model = stacktrm2_st(
            train_opt=opt.train,
            model_opt=opt.model,
            vqvae_opt=opt.vqvae,
            tokenizer_opt=opt.tokenizer,
            generation_opt=opt.generation,
            load_vqvae=load_vqvae
        )
    elif opt.model.name.lower() in ['stacktrm2_sharecb']:
        log.info(f"Start to load model: {opt.model.name}...")
        model = stacktrm2_shareCB(
            train_opt=opt.train,
            model_opt=opt.model,
            vqvae_opt=opt.vqvae,
            tokenizer_opt=opt.tokenizer,
            generation_opt=opt.generation,
            load_vqvae=load_vqvae
        )
    elif opt.model.name.lower() in ['stacktrm2_sharecb_vp']:
        log.info(f"Start to load model: {opt.model.name}...")
        model = stacktrm2_shareCB_VP(
            train_opt=opt.train,
            model_opt=opt.model,
            vqvae_opt=opt.vqvae,
            tokenizer_opt=opt.tokenizer,
            generation_opt=opt.generation,
            load_vqvae=load_vqvae
        )
    elif opt.model.name.lower() in ['stacktrm2_sharecb_vp_wd_fixpremo']:
        log.info(f"Start to load model: {opt.model.name}...")
        model = stacktrm2_shareCB_VP_wd_fixpremo(
            train_opt=opt.train,
            model_opt=opt.model,
            vqvae_opt=opt.vqvae,
            tokenizer_opt=opt.tokenizer,
            generation_opt=opt.generation,
            load_vqvae=load_vqvae
        )
    elif opt.model.name.lower() in ['stacktrm2_sharecb_vp_wd_fixpremo_noise']:
        log.info(f"Start to load model: {opt.model.name}...")
        model = stacktrm2_shareCB_VP_wd_fixpremo_noise(
            train_opt=opt.train,
            model_opt=opt.model,
            vqvae_opt=opt.vqvae,
            tokenizer_opt=opt.tokenizer,
            generation_opt=opt.generation,
            load_vqvae=load_vqvae
        )
    elif opt.model.name.lower() in ['stacktrm2_sharecb_vp_wd_fixpremo_mo']:
        log.info(f"Start to load model: {opt.model.name}...")
        model = stacktrm2_shareCB_VP_wd_fixpremo_mo(
            train_opt=opt.train,
            model_opt=opt.model,
            vqvae_opt=opt.vqvae,
            tokenizer_opt=opt.tokenizer,
            generation_opt=opt.generation,
            load_vqvae=load_vqvae
        )
    elif opt.model.name.lower() in ['stacktrm2_sharecb_vp_wd_fixpremo_moco']:
        log.info(f"Start to load model: {opt.model.name}...")
        model = stacktrm2_shareCB_VP_wd_fixpremo_moco(
            train_opt=opt.train,
            model_opt=opt.model,
            vqvae_opt=opt.vqvae,
            tokenizer_opt=opt.tokenizer,
            generation_opt=opt.generation,
            load_vqvae=load_vqvae
        )
    elif opt.model.name.lower() in ['stacktrm2_sharecb_vp_wd']:
        log.info(f"Start to load model: {opt.model.name}...")
        model = stacktrm2_shareCB_VP_wd(
            train_opt=opt.train,
            model_opt=opt.model,
            vqvae_opt=opt.vqvae,
            tokenizer_opt=opt.tokenizer,
            generation_opt=opt.generation,
            load_vqvae=load_vqvae
        )
    elif opt.model.name.lower() in ['stacktrm2_sharecb_vp_wd_noise']:
        log.info(f"Start to load model: {opt.model.name}...")
        model = stacktrm2_shareCB_VP_wd_noise(
            train_opt=opt.train,
            model_opt=opt.model,
            vqvae_opt=opt.vqvae,
            tokenizer_opt=opt.tokenizer,
            generation_opt=opt.generation,
            load_vqvae=load_vqvae
        )
    elif opt.model.name.lower() in ['stacktrm_vqemb']:
        log.info(f"Start to load model: {opt.model.name}...")
        model = stacktrm_vqemb(
            train_opt=opt.train,
            model_opt=opt.model,
            vqvae_opt=opt.vqvae,
            tokenizer_opt=opt.tokenizer,
            generation_opt=opt.generation,
            load_vqvae=load_vqvae
        )
    elif opt.model.name.lower() in ['base_ds']:
        log.info(f"Start to load model: {opt.model.name}...")
        model = Base_DS(
            train_opt=opt.train,
            model_opt=opt.model,
            vqvae_opt=opt.vqvae,
            tokenizer_opt=opt.tokenizer,
            generation_opt=opt.generation,
            load_vqvae=load_vqvae
        )
    elif opt.model.name.lower() in ['base_ds_class']:
        log.info(f"Start to load model: {opt.model.name}...")
        model = Base_DS_Class(
            train_opt=opt.train,
            model_opt=opt.model,
            vqvae_opt=opt.vqvae,
            tokenizer_opt=opt.tokenizer,
            generation_opt=opt.generation,
            load_vqvae=load_vqvae
        )
    else:
        raise NotImplementedError(f"Model: {opt.model.name.lower()}")

    return model, model.start_step, model.start_epoch