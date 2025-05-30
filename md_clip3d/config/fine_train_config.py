import os
from easydict import EasyDict as edict
from md_clip3d.utils.clip_helpers import FixedNormalizer


__C = edict()
cfg = __C

# root path of your project
project_path = "/user/HCL-AL"

##################################
# general parameters
##################################

__C.general = {}

# image-label pair list
# training csv file, head format
# single modality [image_path, class, x, y, z, width, height, depth]
# multi modality [image_path, image_path1, ..... , class, x, y, z, width, height, depth]
__C.general.im_clip_list = os.path.join(project_path, "data", "train_fine.csv")
__C.general.location_json = os.path.join(project_path, "md_clip3d", "library", "anatomy_vocabulary_library.json")
__C.general.translate_json = os.path.join(project_path, "md_clip3d", "library", "augmented_location_descriptions.json")

# the output of training models and logs
__C.general.save_dir = 'models'

# continue training from certain epoch, -1 to train from scratch
__C.general.resume_epoch = -1

# the number of GPUs used in training
__C.general.num_gpus = 1

# random seed used in training (debugging purpose), -1 for random
__C.general.seed = -1


##################################
# data set parameters
##################################

__C.dataset = {}

__C.dataset.coarse_header = 'coarse_gt'
__C.dataset.fine_header = 'fine_gt'
__C.dataset.related_header = 'coarse_labels'

# the number of input channels
__C.dataset.input_channel = 3

# crop intensity normalizers (to [-1, 1])
# one normalizer corresponds to one input modality
# 1) FixedNormalizer: use fixed mean and standard deviation to normalize intensity
# 2) AdaptiveNormalizer: use minimum and maximum intensity of crop to normalize intensity
__C.dataset.crop_normalizers = [FixedNormalizer(mean=40, stddev=200, clip=True),
                                FixedNormalizer(mean=3, stddev=3, clip=False),
                                FixedNormalizer(mean=0, stddev=1, clip=False)]


__C.dataset.related_rate = 0.8
__C.dataset.case_per_epoch = 500

# sample frequency of words from different regions
__C.dataset.region_sample_frequency = [1, 2, 3, 2, 1, 1, 0.5, 0.5]

# interpolation method(s) for each input channel
__C.dataset.sample_method = 'fixed_length'

# interpolation method(s) for each input channel
__C.dataset.interpolation = ['linear', 'linear', 'nn']

# input voxel size (w, h, d)
__C.dataset.crop_size = [64, 64, 64]

# index of channel(s) to process disturb lesion
__C.dataset.lesion_idx = [2]

# random flip input crop
__C.dataset.random_flip = False

# random rotate input crop in degrees
__C.dataset.rotate_config = {'rot_prob': 0.5, 'rot_angle_degree': 10, 'rot_axis': None}

# random scale ratio of input crop
__C.dataset.scale_config = {'scale_prob': 0.5, 'scale_min_ratio': 0.8, 'scale_max_ratio': 1.2, 'scale_isotropic': True}

# parameter for fixed_length
# box_center_random (unit: mm)
__C.dataset.spacing = [1.5, 1.5, 1.5]
__C.dataset.box_center_random = [3, 3, 3]
__C.dataset.box_percent_padding = 0.75  # nouse


#####################################
# net
#####################################
__C.net = {}
__C.net.pretrained_model_dir = os.path.join(project_path, "pretrained_models")
__C.net.name = 'ClipText_ResidualNet'

# 锟角否冻斤拷 text encoder
__C.net.frozen_text_encoder = True

######################################
# training parameters
######################################
__C.train = {}
__C.train.epochs = 301
__C.train.batchsize = 64
__C.train.num_threads = 16
__C.train.lr = 1e-4


####################################
# training loss
####################################

__C.loss = {}

####################################
# loss 
####################################
__C.loss.name = 'SkipTopNClip'
__C.loss.params = {"class_num": int(__C.train.batchsize/__C.general.num_gpus), "label_smooth": 0.1, "skip_num": 5}


####################################
# CosineAnnealing Args: T_max,eta_min,last_epoch
# Step            Args: step_size, gamma, last_epoch
# MultiStep       Args: milestones, gamma, last_epoch
# Exponential     Args: gamma, last_epoch
####################################
__C.train.lr_scheduler = {}
__C.train.lr_scheduler.name = "MultiStep"
__C.train.lr_scheduler.params = {"milestones": [100, 200], "gamma": 0.1, "last_epoch": -1}


####################################
# Adam           Args: betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False
# SGD            Args: momentum=0, dampening=0, weight_decay=0, nesterov=False
####################################
__C.train.optimizer = {}
__C.train.optimizer.name = "Adam"
__C.train.optimizer.params = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 5e-4, "amsgrad": False}

# the number of batches to update loss curve
__C.train.plot_snapshot = 100

# the number of batches to save model
__C.train.save_epochs = 10

