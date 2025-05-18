from __future__ import print_function
import os
import importlib

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn

from md_clip3d.utils.base.dilate import imdilate
from md_clip3d.utils.base.model_io import load_pytorch_model
from md_clip3d.utils.clip_helpers import image_crop, FixedNormalizer
from md_clip3d.utils.clip_utils import *

def net_classify(images, text_tokens, model, topk=None, no_sort=False):
   """ clip core function
   :param images: image object list to be classified
   :param model: loaded segmentation model
   :return: clip result
   """
   if not topk:
      topk = int(text_tokens.size(0))

   net = model['net']

   iso_images = images
   if model['crop_normalizers'] is not None:
      for idx in range(len(images)):
         iso_images[idx] = model['crop_normalizers'][idx](iso_images[idx])
   
   for idx in range(len(iso_images)):
      img_npy = sitk.GetArrayFromImage(iso_images[idx])
      iso_images[idx] = np.transpose(img_npy, (2, 1, 0))
   

   with torch.no_grad():
      iso_images = torch.stack([torch.from_numpy(iso_image) for iso_image in iso_images])
      iso_image = (iso_images.unsqueeze(0)).cuda()
      text_tokens = text_tokens.cuda()

      # Encode images and text
      image_features = net.module.encode_image(iso_image).float()
      text_features = net.module.encode_text(text_tokens).float()

      # Compute image-text feature similarity and retrieve TopK labels with probability values
      image_features /= image_features.norm(dim=-1, keepdim=True)
      text_features /= text_features.norm(dim=-1, keepdim=True)

      text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
      if no_sort:
         top_probs = text_probs.cpu()
         top_labels = torch.tensor(range(topk)).unsqueeze(0).cpu()
      else:
         top_probs, top_labels = text_probs.cpu().topk(topk, dim=-1)

   return top_probs, top_labels

def load_model(model_dir, pretrained_model_dir):
   param_file = os.path.join(model_dir, 'params.pth')
   
   if not os.path.isfile(param_file):
      print('{} param file not found'.format(model_dir))
      return None

   # load network parameters
   state = load_pytorch_model(param_file)

   # load network structure
   net_name = state['net']
   net_module = importlib.import_module('md_clip3d.network.' + net_name)

   net = net_module.ClipNet(state['input_channel'], state['crop_size'], pretrained_model_dir)
   net = nn.parallel.DataParallel(net)
   net = net.cuda()
   net.load_state_dict(state['state_dict'])
   net.eval()

   crop_normalizers = []
   for crop_normalizer in state['crop_normalizers']:
      if crop_normalizer['type'] == 0:
         crop_normalizers.append(FixedNormalizer(
            crop_normalizer['mean'], crop_normalizer['stddev'], crop_normalizer['clip']))
      else:
         raise ValueError('unknown normalizer type')

   model_dict = {
      'net': net,
      'spacing': state['spacing'],
      'max_stride': state['max_stride'],
      'crop_size':  state['crop_size'],
      'crop_normalizers': crop_normalizers,
      'box_percent_padding': state['box_percent_padding'],
      'sample_method': state['sample_method'],
      'interpolation': state['interpolation'],
      'input_channel': state['input_channel'],
      'lesion_idx': state['lesion_idx']
   }

   return model_dict, net_name

def classify_load_model(model_folder, pretrained_model_dir, gpu_id=0):
   assert isinstance(gpu_id, int)

   # switch to specific gpu
   os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_id)
   assert torch.cuda.is_available(), 'CUDA is not available! Please check nvidia driver!'

   model = dict()
   model['clip'], net_name = load_model(model_folder, pretrained_model_dir)
   model['gpu_id'] = gpu_id

   # switch back to default
   del os.environ['CUDA_VISIBLE_DEVICES']

   return model, net_name

def load_crop(images, model, bbox, target_label=0):
   """
   load crop images using the pre-loaded model
   :param images:              an image object list
   :param model:              the pre-loaded model
   :param center:             the object center
   :param size:               the object size
   :param verbose             whether to print runtime information
   :return: a segmentation image
   """
   if isinstance(images, sitk.Image):
      images = [images]
   elif not isinstance(images, list):
      raise ValueError("images must be sitk.Image or a list of sitk.Image")

   # Switch to specific GPU
   os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(model['gpu_id'])
   assert torch.cuda.is_available(), 'CUDA is not available! Please check nvidia driver!'

   # load and preprocess the image,
   for i in range(len(images)):
      if images[i].GetPixelID() != sitk.sitkFloat32:
         images[i] = sitk.Cast(images[i], sitk.sitkFloat32)

   crop_size = model['clip']['crop_size']
   center, _ = bbox[0:3], bbox[3:6]

   images_forward = list()
   if model['clip']['sample_method'] == 'fixed_length':
      spacing = model['clip']['spacing']
      for idx in range(len(images)):
         images_forward.append(image_crop(
            images[idx], center, spacing, crop_size, method=model['clip']['interpolation'][idx].lower()))
   else:
      raise ValueError("Unknown sample method!")

   # Retain only the lesion-centered mask
   for idx in model['clip']['lesion_idx']:
      if not target_label:
         _, images_forward[idx] = label_connected_component(images_forward[idx], connectivity=26)
         target_label, labels = search_label_by_bbox(images_forward[idx], bbox)
      if target_label:
         images_forward[idx] = convert_multi_label_to_binary(images_forward[idx], target_label)
         continue
      crop_lesion = image_crop(
         images[idx], center, [s/2 for s in spacing], [c*2 for c in crop_size], method=model['clip']['interpolation'][idx].lower())
      _, crop_lesion = label_connected_component(crop_lesion, connectivity=26)
      target_label, labels = search_label_by_bbox(crop_lesion, bbox)
      if target_label:
         crop_lesion = convert_multi_label_to_binary(crop_lesion, target_label)
         crop_lesion = imdilate(crop_lesion, 1, 1, connectivity=6)
         crop_lesion = image_crop(crop_lesion, center, spacing, crop_size, method='nn')
         images_forward[idx] = crop_lesion
         continue
      print(f'Warning: Cannot match bbox and lesion mask!')
      images_forward[idx] = convert_labels_to_target_label(images_forward[idx], labels, 0)

   return images_forward