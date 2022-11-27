import sys
sys.path.append('../..')
from src.utils import get_logger



import torch
try:
    from src.model.MoCoVQVAE.VQVAE import VQVAEModel as moco
    from src.model.MoCoVQVAE_DIFF.VQVAE import VQVAEModel as moco_diff
    from src.model.MoCoVQVAE_woPA.VQVAE import VQVAEModel as moco_wopa
    from src.model.MoCoVQVAE_wCD.VQVAE import VQVAEModel as moco_wcd
    from src.model.MoCoVQVAE_wCD_shareCB.VQVAE import VQVAEModel as moco_wcd_sharecb
    from src.model.MoCoVQVAEwCDsCB_mo.VQVAE import VQVAEModel as moco_wcd_scb_mo
    from src.model.MoCoVQVAEwCDsCB_como.VQVAE import VQVAEModel as moco_wcd_scb_como
    from src.model.MoCoVQVAEwCDsCB_como2.VQVAE import VQVAEModel as moco_wcd_scb_como2
    from src.model.MoCoVQVAE_wDISC.VQVAE import VQVAEModel as moco_wdisc
    from src.model.MoCoVQVAE_woPA_AugCB.VQVAE import VQVAEModel as moco_augcb
except:
    from MoCoVQVAE.src.model.MoCoVQVAE.VQVAE import VQVAEModel as moco
    from MoCoVQVAE.src.model.MoCoVQVAE_DIFF.VQVAE import VQVAEModel as moco_diff
    from MoCoVQVAE.src.model.MoCoVQVAE_woPA.VQVAE import VQVAEModel as moco_wopa
    from MoCoVQVAE.src.model.MoCoVQVAE_wCD.VQVAE import VQVAEModel as moco_wcd
    from MoCoVQVAE.src.model.MoCoVQVAE_wCD_shareCB.VQVAE import VQVAEModel as moco_wcd_sharecb
    from MoCoVQVAE.src.model.MoCoVQVAEwCDsCB_como.VQVAE import VQVAEModel as moco_wcd_scb_como
    from MoCoVQVAE.src.model.MoCoVQVAEwCDsCB_mo.VQVAE import VQVAEModel as moco_wcd_scb_mo
    from MoCoVQVAE.src.model.MoCoVQVAEwCDsCB_como2.VQVAE import VQVAEModel as moco_wcd_scb_como2
    from MoCoVQVAE.src.model.MoCoVQVAE_wDISC.VQVAE import VQVAEModel as moco_wdisc
    from MoCoVQVAE.src.model.MoCoVQVAE_woPA_AugCB.VQVAE import VQVAEModel as moco_augcb

def get_model(opt):
    model_opt = opt['model']

    Logger = get_logger()
    Logger.info(f"Start to create model {model_opt['name']}")

    model_name = model_opt['name'].lower()
    if model_name == 'mocovqvae_wcd':
        model = moco_wcd(model_opt, opt)
    elif model_name == 'mocovqvae_wcd_sharecb':
        model = moco_wcd_sharecb(model_opt, opt)
    elif model_name == 'mocovqvaewcdscb_mo':
        model = moco_wcd_scb_mo(model_opt, opt)
    elif model_name == 'mocovqvaewcdscb_como':
        model = moco_wcd_scb_como(model_opt, opt)
    elif model_name == 'mocovqvaewcdscb_como2':
        model = moco_wcd_scb_como2(model_opt, opt)
    elif model_name == 'mocovqvae_diff':
        model = moco_diff(num_hiddens=model_opt['num_hiddens'],
                          num_residual_layers=model_opt['num_residual_layers'],
                          num_residual_hiddens=model_opt['num_residual_hiddens'],
                          embedding_dim_c=model_opt['embedding_dim_c'],
                          embedding_dim_m=model_opt['embedding_dim_m'],
                          num_embeddings_c=model_opt['num_embeddings_c'],
                          num_embeddings_m=model_opt['num_embeddings_m'],
                          ds_motion=model_opt['ds_motion'],
                          ds_content=model_opt['ds_content'],
                          num_head=model_opt['num_head'],
                          num_group=model_opt['num_group'],
                          num_frames=opt['dataset']['num_frames'],
                          suf_method=model_opt['suf_method'],
                          decoder_type=model_opt['decoder_type'],
                          encoder_mo_type=model_opt['encoder_mo_type'],
                          commitment_cost=model_opt['commitment_cost'],
                          decay=model_opt['decay'],
                          augcb=model_opt['if_augcb'],
                          with_lpips=model_opt['with_lpips'],
                          lpips_factor=model_opt['lpips_factor'],
                          ABS_weight=model_opt['ABS_weight'],
                          MSE_weight=model_opt['MSE_weight'],
                          Gen_weight=model_opt['Gen_weight'],
                          disc_name=model_opt['disc_name'],
                          disc_opt=model_opt['disc_opt'],
                          disc_start_step=opt['train']['disc_start_step'])
    elif model_name == 'mocovqvae_wopa':
        model = moco_wopa(num_hiddens=model_opt['num_hiddens'],
                      num_residual_layers=model_opt['num_residual_layers'],
                      num_residual_hiddens=model_opt['num_residual_hiddens'],
                      embedding_dim_c=model_opt['embedding_dim_c'],
                      embedding_dim_m=model_opt['embedding_dim_m'],
                      num_embeddings_c=model_opt['num_embeddings_c'],
                      num_embeddings_m=model_opt['num_embeddings_m'],
                      ds_motion=model_opt['ds_motion'],
                      ds_content=model_opt['ds_content'],
                      num_head=model_opt['num_head'],
                      num_group=model_opt['num_group'],
                      num_frames=opt['dataset']['num_frames'],
                      suf_method=model_opt['suf_method'],
                      decoder_type=model_opt['decoder_type'],
                      encoder_mo_type=model_opt['encoder_mo_type'],
                      commitment_cost=model_opt['commitment_cost'],
                      decay=model_opt['decay'],
                      with_lpips=model_opt['with_lpips'],
                      lpips_factor=model_opt['lpips_factor'])
    elif model_name == 'mocovqvae_augcb':
        model = moco_augcb(num_hiddens=model_opt['num_hiddens'],
                           num_residual_layers=model_opt['num_residual_layers'],
                           num_residual_hiddens=model_opt['num_residual_hiddens'],
                           embedding_dim_c=model_opt['embedding_dim_c'],
                           embedding_dim_m=model_opt['embedding_dim_m'],
                           num_embeddings_c=model_opt['num_embeddings_c'],
                           num_embeddings_m=model_opt['num_embeddings_m'],
                           ds_motion=model_opt['ds_motion'],
                           ds_content=model_opt['ds_content'],
                           num_head=model_opt['num_head'],
                           num_group=model_opt['num_group'],
                           num_frames=opt['dataset']['num_frames'],
                           suf_method=model_opt['suf_method'],
                           decoder_type=model_opt['decoder_type'],
                           encoder_mo_type=model_opt['encoder_mo_type'],
                           commitment_cost=model_opt['commitment_cost'],
                           decay=model_opt['decay'],
                           with_lpips=model_opt['with_lpips'],
                           lpips_factor=model_opt['lpips_factor'])
    elif model_name == 'mocovqvae_wid' or model_name == 'mocovqvae_wcd' or model_name == 'mocovqvae_wicd':
        model = moco_wdisc(num_hiddens=model_opt['num_hiddens'],
                          num_residual_layers=model_opt['num_residual_layers'],
                          num_residual_hiddens=model_opt['num_residual_hiddens'],
                          embedding_dim_c=model_opt['embedding_dim_c'],
                          embedding_dim_m=model_opt['embedding_dim_m'],
                          num_embeddings_c=model_opt['num_embeddings_c'],
                          num_embeddings_m=model_opt['num_embeddings_m'],
                          ds_motion=model_opt['ds_motion'],
                          ds_content=model_opt['ds_content'],
                          num_head=model_opt['num_head'],
                          num_group=model_opt['num_group'],
                          num_frames=opt['dataset']['num_frames'],
                          suf_method=model_opt['suf_method'],
                          decoder_type=model_opt['decoder_type'],
                          encoder_mo_type=model_opt['encoder_mo_type'],
                          commitment_cost=model_opt['commitment_cost'],
                          decay=model_opt['decay'],
                          augcb=model_opt.get('if_augcb', 0),
                          with_lpips=model_opt['with_lpips'],
                          lpips_factor=model_opt['lpips_factor'],
                          ABS_weight=model_opt['ABS_weight'],
                          MSE_weight=model_opt['MSE_weight'],
                          Gen_weight=model_opt.get('Gen_weight', 0),
                          disc_name=model_opt.get('disc_name', 'patchwise'),
                          disc_opt=model_opt.get('disc_opt', {}),
                          disc_start_step=opt['train']['disc_start_step'])
    else:
        raise ValueError(f"No implemention for model {model_name}")

    # resume ckpt
    start_step = 0
    if model_opt['checkpoint_path'] is not None:
        state = torch.load(model_opt['checkpoint_path'], map_location='cpu')
        start_step = state['steps']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state['state'].items():
            if 'total_ops' in k or 'total_params' in k:
                continue
            if 'perceptual_loss' in k or '_discriminator' in k:
            # if 'perceptual_loss' in k:
                continue
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v

        # model.load_state_dict(new_state_dict, strict=False)
        model.load_state_dict(new_state_dict, strict=model_opt['load_strict'])
        Logger.info("Successfully load state {} with step {}.".format(model_opt['checkpoint_path'], start_step))

    elif model_opt['pretrain_path'] is not None:
        state = torch.load(model_opt['pretrain_path'], map_location='cpu')
        start_step = 0

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state['state'].items():
            if 'total_ops' in k or 'total_params' in k:
                continue
            if 'perceptual_loss' in k or '_discriminator' in k:
            # if 'perceptual_loss' in k:
                continue
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v

        model.load_state_dict(new_state_dict, strict=model_opt['load_strict'])
        Logger.info("Successfully load state {} with step {}.".format(model_opt['pretrain_path'], start_step))

    return model, start_step