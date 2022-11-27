import sys
sys.path.append('../..')
from src.utils import get_logger

from .MSRVTTDataset import MsrvttFramesDataset
from .VideoFramesDataset import VideoFramesDataset
from .VideoDiffFramesDataset import VideoDiffFramesDataset
from .VideoDiffFramesDataset import VideoDiffFramesDataset_wRP
from .VideoDiffFramesDataset import VideoDiffFramesDataset_woCP
from .VideoDiffFramesDataset_General import VideoDiffFramesDataset_General
from .VideoDiffFramesDataset_EXT import VideoDiffFramesDataset_EXT
from .VideoDiffFramesDataset_EXT import VideoDiffFramesDataset_wRP_EXT
from .VideoDiffFramesDataset_EXT import VideoDiffFramesDataset_woCP_EXT
from .VideoDiffFramesDataset_EXT_FirstFrame import VideoDiffFramesDataset_EXT_FirstFrame
from .VideoDiffFramesDataset_EXT_TwoFrames import VideoDiffFramesDataset_EXT_FirstTwoFrames
from .VideoDiffFramesDataset_EXT_FirstKFrames import VideoDiffFramesDataset_EXT_FirstKFrames
from .VideoDiffFramesDataset_EXT_FirstKFrames import VideoDiffFramesDataset_woCP_EXT_FirstKFrames
from .VideoDiffFramesDataset_woFixPre import VideoDiffFramesDataset_woFixPre
from .VideoDiffFramesDataset_FixPreTo0 import VideoDiffFramesDataset_FixPreTo0
from .VideoDiffFramesDataset_FullBG import VideoDiffFramesDataset_FullBG
from .VideoDiffFramesDataset_MoCo import VideoDiffFramesDataset_MoCo
from .VideoDiffFramesDataset_fromnpy import VideoDiffFramesDataset_fromnpy
from .VideoDiffFramesDataset_FullBGID import VideoDiffFramesDataset_FullBGID
from .VideoDiffFramesDataset_FullBGID_woFixPre import VideoDiffFramesDataset_FullBGID_woFixPre


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def get_dataset(opt):
    data_opt = opt['dataset']
    dataset_name = data_opt['name'].lower()

    Logger = get_logger()
    Logger.info(f"Start to load dataset {dataset_name}...")

    if dataset_name == 'msrvtt':
        cur_opt = data_opt['train']
        trainset = MsrvttFramesDataset(datapath=cur_opt['datapath'],
                                       idspath=cur_opt['idspath'],
                                       img_size=cur_opt['img_size'])

        cur_opt = data_opt['val']
        testset = MsrvttFramesDataset(datapath=cur_opt['datapath'],
                                       idspath=cur_opt['idspath'],
                                       img_size=cur_opt['img_size'])

    elif dataset_name == 'videodiffframes_como':
        trainset = VideoDiffFramesDataset_MoCo(datapath=data_opt['train']['datapath'],
                                          idspath=data_opt['train']['idspath'],
                                          used_fps=data_opt['train']['usedfps'],
                                          limit=data_opt['limit'],
                                          img_size=data_opt['img_size'],
                                          num_frames=data_opt['num_frames'])

        testset = VideoDiffFramesDataset_MoCo(datapath=data_opt['val']['datapath'],
                                         idspath=data_opt['val']['idspath'],
                                         used_fps=data_opt['val']['usedfps'],
                                         limit=data_opt['limit'],
                                         img_size=data_opt['img_size'],
                                         num_frames=data_opt['num_frames'])

    elif dataset_name == 'videodiffframes_general':
        trainset = VideoDiffFramesDataset_General(datapath=data_opt['train']['datapath'],
                                                  img_idspath=data_opt['train']['img_idspath'],
                                                  vid_idspath=data_opt['train']['vid_idspath'],
                                                  img_duplicate=data_opt['train']['img_duplicate'],
                                                  vid_duplicate=data_opt['train']['vid_duplicate'],
                                                  limit=data_opt['limit'],
                                                  img_size=data_opt['img_size'],
                                                  num_frames=data_opt['num_frames'])

        testset = VideoDiffFramesDataset_General(datapath=data_opt['val']['datapath'],
                                                  img_idspath=data_opt['val']['img_idspath'],
                                                  vid_idspath=data_opt['val']['vid_idspath'],
                                                  img_duplicate=data_opt['val']['img_duplicate'],
                                                  vid_duplicate=data_opt['val']['vid_duplicate'],
                                                  limit=data_opt['limit'],
                                                  img_size=data_opt['img_size'],
                                                  num_frames=data_opt['num_frames'])

    elif dataset_name == 'videodiffframes':
        trainset = VideoDiffFramesDataset(datapath=data_opt['train']['datapath'],
                                          idspath=data_opt['train']['idspath'],
                                          used_fps=data_opt['train']['usedfps'] if 'usedfps' in data_opt['train'].keys() else None,
                                          limit=data_opt['limit'],
                                          img_size=data_opt['img_size'],
                                          num_frames=data_opt['num_frames'])

        testset = VideoDiffFramesDataset(datapath=data_opt['val']['datapath'],
                                         idspath=data_opt['val']['idspath'],
                                         used_fps=data_opt['val']['usedfps'] if 'usedfps' in data_opt['val'].keys() else None,
                                         limit=data_opt['limit'],
                                         img_size=data_opt['img_size'],
                                         num_frames=data_opt['num_frames'])

        # testset = VideoDiffFramesDataset_EXT(datapath=data_opt['val']['datapath'],
        #                                      idspath=data_opt['val']['idspath'],
        #                                      used_fps=data_opt['val']['usedfps'],
        #                                      limit=data_opt['limit'],
        #                                      img_size=data_opt['img_size'],
        #                                      num_frames=data_opt['num_frames'],
        #                                      ret_mode='list')

    elif dataset_name == 'videodiffframes_wocp':
        trainset = VideoDiffFramesDataset_woCP(datapath=data_opt['train']['datapath'],
                                          idspath=data_opt['train']['idspath'],
                                          used_fps=data_opt['train']['usedfps'] if 'usedfps' in data_opt['train'].keys() else None,
                                          limit=data_opt['limit'],
                                          img_size=data_opt['img_size'],
                                          num_frames=data_opt['num_frames'])

        # testset = VideoDiffFramesDataset_woCP(datapath=data_opt['val']['datapath'],
        #                                  idspath=data_opt['val']['idspath'],
        #                                  used_fps=data_opt['val']['usedfps'] if 'usedfps' in data_opt['val'].keys() else None,
        #                                  limit=data_opt['limit'],
        #                                  img_size=data_opt['img_size'],
        #                                  num_frames=data_opt['num_frames'])

        testset = VideoDiffFramesDataset_woCP_EXT(datapath=data_opt['val']['datapath'],
                                              idspath=data_opt['val']['idspath'],
                                              used_fps=data_opt['val']['usedfps'] if 'usedfps' in data_opt[
                                                  'val'].keys() else None,
                                              limit=data_opt['limit'],
                                              img_size=data_opt['img_size'],
                                              num_frames=data_opt['num_frames'], ret_mode='list')

    elif dataset_name == 'videodiffframes_wrp':
        trainset = VideoDiffFramesDataset_wRP(datapath=data_opt['train']['datapath'],
                                          idspath=data_opt['train']['idspath'],
                                          used_fps=data_opt['train']['usedfps'] if 'usedfps' in data_opt['train'].keys() else None,
                                          limit=data_opt['limit'],
                                          img_size=data_opt['img_size'],
                                          num_frames=data_opt['num_frames'],
                                          save_batch_dir=data_opt['train']['save_batch_dir'])

        testset = VideoDiffFramesDataset_wRP_EXT(datapath=data_opt['val']['datapath'],
                                         idspath=data_opt['val']['idspath'],
                                         used_fps=data_opt['val']['usedfps'] if 'usedfps' in data_opt['val'].keys() else None,
                                         limit=data_opt['limit'],
                                         img_size=data_opt['img_size'],
                                         num_frames=data_opt['num_frames'],
                                         ret_mode='list')



    elif dataset_name == 'videodiffframes_fromnpy':
        trainset = VideoDiffFramesDataset_fromnpy(datapath=data_opt['train']['datapath'])

        testset = VideoDiffFramesDataset_fromnpy(datapath=data_opt['val']['datapath'])

    elif dataset_name == 'videodiffframes_ext':
        trainset = VideoDiffFramesDataset_EXT(datapath=data_opt['train']['datapath'],
                                              idspath=data_opt['train']['idspath'],
                                              used_fps=data_opt['train']['usedfps'],
                                              limit=data_opt['limit'],
                                              img_size=data_opt['img_size'],
                                              num_frames=data_opt['num_frames'],
                                              step=data_opt.get('step', None),
                                              ret_mode=data_opt.get('ret_mode', 'dict'))

        testset = VideoDiffFramesDataset_EXT(datapath=data_opt['val']['datapath'],
                                             idspath=data_opt['val']['idspath'],
                                             used_fps=data_opt['val']['usedfps'],
                                             limit=data_opt['limit'],
                                             img_size=data_opt['img_size'],
                                             num_frames=data_opt['num_frames'],
                                             step=data_opt.get('step', None),
                                             ret_mode=data_opt.get('ret_mode', 'dict'))

    elif dataset_name == 'videodiffframes_wocp_ext':
        trainset = VideoDiffFramesDataset_woCP_EXT(datapath=data_opt['train']['datapath'],
                                              idspath=data_opt['train']['idspath'],
                                              used_fps=data_opt['train']['usedfps'],
                                              limit=data_opt['limit'],
                                              img_size=data_opt['img_size'],
                                              num_frames=data_opt['num_frames'],
                                              step=data_opt.get('step', None),
                                              ret_mode=data_opt.get('ret_mode', 'dict'))

        testset = VideoDiffFramesDataset_woCP_EXT(datapath=data_opt['val']['datapath'],
                                             idspath=data_opt['val']['idspath'],
                                             used_fps=data_opt['val']['usedfps'],
                                             limit=data_opt['limit'],
                                             img_size=data_opt['img_size'],
                                             num_frames=data_opt['num_frames'],
                                             step=data_opt.get('step', None),
                                              ret_mode=data_opt.get('ret_mode', 'dict'))

    elif dataset_name == 'videodiffframes_wrp_ext':
        trainset = VideoDiffFramesDataset_wRP_EXT(datapath=data_opt['train']['datapath'],
                                              idspath=data_opt['train']['idspath'],
                                              used_fps=data_opt['train']['usedfps'],
                                              limit=data_opt['limit'],
                                              img_size=data_opt['img_size'],
                                              num_frames=data_opt['num_frames'],
                                              step=data_opt.get('step', None),
                                              ret_mode=data_opt.get('ret_mode', 'dict'))

        testset = VideoDiffFramesDataset_wRP_EXT(datapath=data_opt['val']['datapath'],
                                             idspath=data_opt['val']['idspath'],
                                             used_fps=data_opt['val']['usedfps'],
                                             limit=data_opt['limit'],
                                             img_size=data_opt['img_size'],
                                             num_frames=data_opt['num_frames'],
                                             step=data_opt.get('step', None),
                                                 ret_mode=data_opt.get('ret_mode', 'dict'))

    elif dataset_name == 'videodiffframes_ext_firstkframes':

        train_step = data_opt.get('step', None)
        trainset = VideoDiffFramesDataset_EXT_FirstKFrames(datapath=data_opt['train']['datapath'],
                                              idspath=data_opt['train']['idspath'],
                                              used_fps=data_opt['train']['usedfps'],
                                              limit=data_opt['limit'],
                                              img_size=data_opt['img_size'],
                                              num_frames=data_opt['num_frames'],
                                              step=train_step, K=data_opt['K'])

        testset = VideoDiffFramesDataset_EXT_FirstKFrames(datapath=data_opt['val']['datapath'],
                                             idspath=data_opt['val']['idspath'],
                                             used_fps=data_opt['val']['usedfps'],
                                             limit=data_opt['limit'],
                                             img_size=data_opt['img_size'],
                                             num_frames=data_opt['num_frames'],
                                             step=train_step, K=data_opt['K'])

    elif dataset_name == 'videodiffframes_wocp_ext_firstkframes':

        train_step = data_opt.get('step', None)
        trainset = VideoDiffFramesDataset_woCP_EXT_FirstKFrames(datapath=data_opt['train']['datapath'],
                                              idspath=data_opt['train']['idspath'],
                                              used_fps=data_opt['train']['usedfps'],
                                              limit=data_opt['limit'],
                                              img_size=data_opt['img_size'],
                                              num_frames=data_opt['num_frames'],
                                              step=train_step, K=data_opt['K'])

        testset = VideoDiffFramesDataset_woCP_EXT_FirstKFrames(datapath=data_opt['val']['datapath'],
                                             idspath=data_opt['val']['idspath'],
                                             used_fps=data_opt['val']['usedfps'],
                                             limit=data_opt['limit'],
                                             img_size=data_opt['img_size'],
                                             num_frames=data_opt['num_frames'],
                                             step=train_step, K=data_opt['K'])


    elif dataset_name == 'videodiffframes_ext_firstframe':
        trainset = VideoDiffFramesDataset_EXT_FirstFrame(datapath=data_opt['train']['datapath'],
                                              idspath=data_opt['train']['idspath'],
                                              used_fps=data_opt['train']['usedfps'],
                                              limit=data_opt['limit'],
                                              img_size=data_opt['img_size'],
                                              num_frames=data_opt['num_frames'])

        testset = VideoDiffFramesDataset_EXT_FirstFrame(datapath=data_opt['val']['datapath'],
                                             idspath=data_opt['val']['idspath'],
                                             used_fps=data_opt['val']['usedfps'],
                                             limit=data_opt['limit'],
                                             img_size=data_opt['img_size'],
                                             num_frames=data_opt['num_frames'])

    elif dataset_name == 'videodiffframes_ext_firsttwoframes':
        trainset = VideoDiffFramesDataset_EXT_FirstTwoFrames(datapath=data_opt['train']['datapath'],
                                              idspath=data_opt['train']['idspath'],
                                              used_fps=data_opt['train']['usedfps'],
                                              limit=data_opt['limit'],
                                              img_size=data_opt['img_size'],
                                              num_frames=data_opt['num_frames'])

        testset = VideoDiffFramesDataset_EXT_FirstTwoFrames(datapath=data_opt['val']['datapath'],
                                             idspath=data_opt['val']['idspath'],
                                             used_fps=data_opt['val']['usedfps'],
                                             limit=data_opt['limit'],
                                             img_size=data_opt['img_size'],
                                             num_frames=data_opt['num_frames'])

    else:
        raise ValueError(f"No implemention for dataset {dataset_name}!!")

    Logger.info(f"dataset {dataset_name} has train data {len(trainset)}, val data {len(testset)}.")

    return trainset, testset
