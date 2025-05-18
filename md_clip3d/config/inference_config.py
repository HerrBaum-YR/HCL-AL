from easydict import EasyDict as edict

__C = edict()
cfg = __C


##################################
# general parameters
##################################

__C.general = {}


# Two inference mode
# 1. mask as input，input csv format：['image_path', 'image_path1', ..., 'lesion_path']
# 2. bbox as input，input csv format：['image_path', 'image_path1', ..., 'x', 'y', 'z', 'width', 'height', 'depth']
__C.general.mode = 'box' # 'mask'
__C.general.input_path = "bbox.csv"
# __C.general.input_path = "mask.csv"

# output csv path
__C.general.output_folder = "result"

# the id of GPU used for inference
__C.general.gpu_id = 1

__C.general.pretrained_model_dir = 'pretrained_models'


##################################
# library parameters
##################################
__C.library = {}

# json path of anatomy vocabulary library
__C.library.inference_json_path = "md_clip3d/config/anatomy_vocabulary_library.json"
# json path of augmented location descriptions
__C.library.translate_json_path = "md_clip3d/config/augmented_location_descriptions.json"


##################################
# model parameters
##################################
__C.coarse = {}

__C.coarse.run = True
__C.coarse.model_dir = "pretrained_weights/coarse"
__C.coarse.topk = 3  # if set to None, the output results will not be truncated

__C.fine = {}
__C.fine.run = True
__C.fine.model_dir = "pretrained_weights/fine"
__C.fine.filter_by_gender = False
__C.fine.filter_by_size = False
__C.fine.fuse_coarse_thres = 0  # merge the coarse and fine probabilities if fine is below this threshold
