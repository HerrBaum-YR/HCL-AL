import os
import sys
import torch
import shutil
import builtins
import numpy as np

from md_clip3d.utils.base.file_tools import load_module_from_disk
from md_clip3d.utils.base.model_io import load_pytorch_model, save_pytorch_model
from md_clip3d.utils.clip_sampler import CoarsePositionBatchSampler, FinePositionBatchSampler
from md_clip3d.utils.clip_dataset import ClipClsDataSet

def save_checkpoint(net, epoch_idx, batch_idx, cfg, config_file, max_stride):
    """
    save model and parameters into a checkpoint file (.pth)
    :param net: the network object
    :param epoch_idx: the epoch index
    :param batch_idx: the batch index
    :param cfg: the configuration object
    :param config_file: the configuration file path
    :param max_stride: the maximum stride of network
    :return: None
    """
    chk_folder = os.path.join(cfg.general.save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx))
    if not os.path.isdir(chk_folder):
        os.makedirs(chk_folder)
    filename = os.path.join(chk_folder, 'params.pth'.format(epoch_idx))

    state = {'epoch':               epoch_idx,
             'batch':               batch_idx,
             'net':                 cfg.net.name,
             'max_stride':          max_stride,
             'state_dict':          net.state_dict(),
             'spacing':             cfg.dataset.spacing,
             'crop_size':           cfg.dataset.crop_size,
             'crop_normalizers':    [normalizer.to_dict() for normalizer in cfg.dataset.crop_normalizers],
             'box_percent_padding': cfg.dataset.box_percent_padding,
             'sample_method':       cfg.dataset.sample_method,
             'interpolation':       cfg.dataset.interpolation,
             'input_channel':       cfg.dataset.input_channel,
             'lesion_idx':          cfg.dataset.lesion_idx}
    save_pytorch_model(state, filename, is_encrypt=False)
    shutil.copy(config_file, os.path.join(chk_folder, 'config.py'))

def load_checkpoint(epoch_idx, net, save_dir):
    """
    load network parameters from directory
    :param epoch_idx: the epoch idx of model to load
    :param net: the network object
    :param save_dir: the save directory
    :return: loaded epoch index, loaded batch index
    """
    chk_file = os.path.join(save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx), 'params.pth')
    if not os.path.isfile(chk_file):
        raise ValueError('checkpoint file not found: {}'.format(chk_file))
    
    print("Loading model:", chk_file)

    state = load_pytorch_model(chk_file)
    net.load_state_dict(state['state_dict'])

    return state['epoch'], state['batch']

def worker_init(worker_idx):
    """
    The worker initialization function takes the worker id (an int in "[0,
    num_workers - 1]") as input and does random seed initialization for each
    worker.
    :param worker_idx: The worker index.
    :return: None.
    """
    MAX_INT = sys.maxsize
    worker_seed = np.random.randint(int(np.sqrt(MAX_INT))) + worker_idx
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)

def load_clip_cfg_file(config_file):
    """
    :param config_file:  configure file path
    :return:
    """
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)
    cfg = load_module_from_disk(config_file)
    cfg = cfg.cfg

    # convert to absolute path since cfg uses relative path
    root_dir = os.path.dirname(config_file)
    cfg.general.im_clip_list = os.path.join(root_dir, cfg.general.im_clip_list)
    cfg.general.save_dir = os.path.join(root_dir, cfg.general.save_dir)

    return cfg

def generate_coarse_training_data_set(cfg):
    """
    :param cfg:            config contain data set information
    :return:               data loader, length of data set
    """
    if cfg.loss.name in ['Clip', 'TopNLabelSmoothingClip', 'SkipTopNClip']:
        mode = 'train'
    elif cfg.loss.name == 'ClipHardNegSample':
        mode = 'train_hardnegsample'

    data_set = ClipClsDataSet(
        im_clip_list=cfg.general.im_clip_list,
        translate_json=cfg.general.translate_json,
        target_header=cfg.dataset.target_header,
        input_channels=cfg.dataset.input_channel,
        crop_size=cfg.dataset.crop_size,
        crop_normalizers=cfg.dataset.crop_normalizers,
        sample_method=cfg.dataset.sample_method,
        interpolation=cfg.dataset.interpolation,
        spacing=cfg.dataset.spacing,
        box_center_random=cfg.dataset.box_center_random,
        box_percent_padding=cfg.dataset.box_percent_padding,
        rotate_config=cfg.dataset.rotate_config,
        scale_config=cfg.dataset.scale_config,
        random_flip=cfg.dataset.random_flip,
        lesion_idx=cfg.dataset.lesion_idx,
        net_name=cfg.net.name,
        pretrained_model_dir=cfg.net.pretrained_model_dir,
        mode=mode)

    batch_sampler = CoarsePositionBatchSampler(
        dataset=data_set, 
        batchsize=cfg.train.batchsize, 
        epochs=cfg.train.epochs - cfg.general.resume_epoch,
        location_json=cfg.general.location_json,
        region_sample_frequency=cfg.dataset.region_sample_frequency,
        target_header=cfg.dataset.target_header,
        region_num_per_batch=cfg.dataset.region_num_per_batch,
        word_iter_num_per_epoch=cfg.dataset.word_iter_num_per_epoch)

    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_sampler=batch_sampler,
        num_workers=cfg.train.num_threads,
        pin_memory=True,
        worker_init_fn=worker_init)

    return data_loader, batch_sampler.batch_per_epoch

def generate_fine_training_data_set(cfg):
    """
    :param cfg:            config contain data set information
    :return:               data loader, length of data set
    """
    if cfg.loss.name in ['Clip', 'TopNLabelSmoothingClip', 'SkipTopNClip']:
        mode = 'train'
    elif cfg.loss.name == 'ClipHardNegSample':
        mode = 'train_hardnegsample'

    data_set = ClipClsDataSet(
        im_clip_list=cfg.general.im_clip_list,
        translate_json=cfg.general.translate_json,
        target_header=cfg.dataset.fine_header,
        input_channels=cfg.dataset.input_channel,
        crop_size=cfg.dataset.crop_size,
        crop_normalizers=cfg.dataset.crop_normalizers,
        sample_method=cfg.dataset.sample_method,
        interpolation=cfg.dataset.interpolation,
        spacing=cfg.dataset.spacing,
        box_center_random=cfg.dataset.box_center_random,
        box_percent_padding=cfg.dataset.box_percent_padding,
        rotate_config=cfg.dataset.rotate_config,
        scale_config=cfg.dataset.scale_config,
        random_flip=cfg.dataset.random_flip,
        lesion_idx=cfg.dataset.lesion_idx,
        net_name=cfg.net.name,
        pretrained_model_dir=cfg.net.pretrained_model_dir,
        mode=mode)

    batch_sampler = FinePositionBatchSampler(
        dataset=data_set, 
        batchsize=cfg.train.batchsize, 
        epochs=cfg.train.epochs - cfg.general.resume_epoch,
        location_json=cfg.general.location_json,
        region_sample_frequency=cfg.dataset.region_sample_frequency,
        coarse_header=cfg.dataset.coarse_header,
        fine_header=cfg.dataset.fine_header,
        related_header=cfg.dataset.related_header,
        related_rate=cfg.dataset.related_rate,
        case_per_epoch=cfg.dataset.case_per_epoch)
    
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_sampler=batch_sampler,
        num_workers=cfg.train.num_threads,
        pin_memory=True,
        worker_init_fn=worker_init)

    return data_loader, batch_sampler.batch_per_epoch

def setup_workshop(cfg):
    """
    :param cfg:  training configure file
    :return:
    """
    if cfg.general.resume_epoch < 0 and os.path.isdir(cfg.general.save_dir):
        sys.stdout.write("Found non-empty save dir.\nType 'yes' to delete, 'no' to continue: ")
        choice = builtins.input().lower()
        if choice == 'yes':
            shutil.rmtree(cfg.general.save_dir)
        elif choice == 'no':
            pass
        else:
            raise ValueError("Please type either 'yes' or 'no'!")

def init_torch_and_numpy(cfg):
    assert torch.cuda.is_available(), 'CUDA is not available! Please check nvidia driver!'
    torch.backends.cudnn.benchmark = True
    if cfg.general.seed != -1:
        np.random.seed(cfg.general.seed)
        torch.manual_seed(cfg.general.seed)
        torch.cuda.manual_seed(cfg.general.seed)

def set_optimizer(cfg, net):
    """
    :param cfg:   training configure file
    :param net:   pytorch network
    :return:
    """
    if cfg.train.optimizer == 'SGD':
        opt = torch.optim.SGD(net.parameters(),
                              lr=cfg.train.lr,
                              momentum=cfg.train.momentum,
                              weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == 'Adam':
        opt = torch.optim.Adam(net.parameters(),
                               lr=cfg.train.lr,
                               betas=cfg.train.betas,
                               weight_decay=cfg.train.weight_decay)
    else:
        raise ValueError('Unknown loss optimizer')

    return opt
