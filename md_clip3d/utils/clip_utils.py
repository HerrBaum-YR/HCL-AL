import math
import random
import numpy as np
from scipy.ndimage import label
import SimpleITK as sitk
from typing import Union, List

def count_labels(image: sitk.Image, max_labels: int=2048):
   image_npy = sitk.GetArrayFromImage(image)
   data = image_npy.flatten()
   valid_labels = data[(data > 0) & (data < max_labels)]
   unique_labels = np.unique(valid_labels).astype(int)
   has_label = np.zeros(max_labels, dtype=bool)
   has_label[unique_labels] = True
   return np.flatnonzero(has_label).tolist()

def label_connected_component(image: sitk.Image, connectivity: int=6):
   if connectivity not in [6, 26]:
      raise ValueError("Invalid argument: connectivity has to be 6 or 26")
   
   image_npy = sitk.GetArrayFromImage(image).astype(np.int32)
   if connectivity == 6:
      structure = np.ones((3, 3, 3), dtype=int)
      structure[1, 1, 1] = 0
   else:
      structure = np.ones((3, 3, 3), dtype=int)

   labeled_image_npy, num_components = label(image_npy, structure=structure)
   labeled_image = sitk.GetImageFromArray(labeled_image_npy)
   labeled_image.CopyInformation(image)
   return num_components, labeled_image

def convert_labels_to_target_label(mask: sitk.Image, labels: Union[List[int], np.ndarray], target_label: int):
   mask_npy = sitk.GetArrayFromImage(mask)
   if isinstance(labels, np.ndarray):
      labels = labels.tolist()

   set_labels = set(labels)
   labels_contain_zero = 0 in set_labels
   replace_mask = np.isin(mask_npy, list(set_labels))
   if not labels_contain_zero:
      zero_mask = (mask_npy != 0)
      replace_mask = replace_mask & zero_mask

   mask_npy[replace_mask] = target_label
   
   result = sitk.GetImageFromArray(mask_npy)
   result.CopyInformation(mask)
   return result

def convert_multi_label_to_binary(mask: sitk.Image, target_label: int):
   mask_npy = sitk.GetArrayFromImage(mask)
   binary_npy = (mask_npy == target_label).astype(np.int32)
   binary_mask = sitk.GetImageFromArray(binary_npy)
   binary_mask.CopyInformation(mask)
   return binary_mask

def bounding_box_voxel(image: sitk.Image, minVal: float, maxVal: float):
   image_npy = sitk.GetArrayFromImage(image)
   image_npy = np.transpose(image_npy, (2, 1, 0))
   mask = (image_npy >= minVal) & (image_npy <= maxVal)
   if not np.any(mask):
      return None, None

   nonzero_indices = np.argwhere(mask)
   mincorner = tuple(nonzero_indices.min(axis=0))
   maxcorner = tuple(nonzero_indices.max(axis=0))
   return mincorner, maxcorner

def world_box_full(image: sitk.Image, min_voxel: tuple, max_voxel: tuple):
   origin = np.array(image.GetOrigin())
   spacing = np.array(image.GetSpacing())
   direction = np.array(image.GetDirection()).reshape(3,3)

   box_origin = origin + direction @ (np.array(min_voxel) * spacing)
   expanded_max = origin + direction @ (np.array(tuple(np.array(max_voxel) + 1)) * spacing)
   box_size = np.abs(expanded_max - box_origin)
   return tuple(box_origin.tolist()), tuple(box_size.tolist())

def calc_lesion_bbox(mask: sitk.Image, label: int):
   min_voxel, max_voxel = bounding_box_voxel(mask, label, label)
   min_world, size = world_box_full(mask, min_voxel, max_voxel)
   min_world = np.array(min_world)
   size = np.array(size)
   max_world = min_world + size
   center = (min_world + max_world) / 2
   return center, size

def box_iou(center_box1, center_box2):
   box1_min = center_box1[0:3] - center_box1[3:] / 2
   box1_max = center_box1[0:3] + center_box1[3:] / 2
   box2_min = center_box2[0:3] - center_box2[3:] / 2
   box2_max = center_box2[0:3] + center_box2[3:] / 2
   xi1 = max(box1_min[0], box2_min[0])
   yi1 = max(box1_min[1], box2_min[1])
   zi1 = max(box1_min[2], box2_min[2])

   xi2 = min(box1_max[0], box2_max[0])
   yi2 = min(box1_max[1], box2_max[1])
   zi2 = min(box1_max[2], box2_max[2])

   weight = max(xi2 - xi1, 0)
   height = max(yi2 - yi1, 0)
   depth = max(zi2 - zi1, 0)
   inter_area = weight * height * depth
   if inter_area == 0:
      return 0
   box1_area = (box1_max[0] - box1_min[0]) * (box1_max[1] - box1_min[1]) * (box1_max[2] - box1_min[2])
   box2_area = (box2_max[0] - box2_min[0]) * (box2_max[1] - box2_min[1]) * (box2_max[2] - box2_min[2])
   assert box1_area > 0 and box2_area > 0
   union_area = box1_area + box2_area - inter_area
   return inter_area / union_area

def search_label_by_bbox(mask: sitk.Image, bbox_info: np.ndarray, iou_thres: float=0):
   target_label = 0
   mask_npy = sitk.GetArrayFromImage(mask)
   labels = list(np.unique(mask_npy))
   if labels.count(0):
      labels.remove(0)
   iou_max = iou_thres
   for label in labels:
      tmp_center, tmp_size = calc_lesion_bbox(mask, label)
      tmp_bbox = np.concatenate((tmp_center, tmp_size))
      iou = box_iou(np.array(bbox_info), tmp_bbox)
      if iou > iou_max:
         iou_max = iou
         target_label = label
   return target_label, labels

def uniform_sample_point_from_unit_sphere(num_samples=1):
   """
   uniformly sample a point from a unit sphere
   If the number of samples is large, this sampling strategy ensures the uniformity of point distribution on sphere
   :return: a list of 3D points (np.array)
   """
   assert num_samples >= 1, 'number of samples must be >= 1'

   sample_points = np.empty((num_samples, 3), dtype=np.double)
   for i in range(num_samples):
      theta = 2 * math.pi * random.random()
      phi = math.acos(2 * random.random() - 1.0)
      sample_points[i, 0] = math.cos(theta) * math.sin(phi)
      sample_points[i, 1] = math.sin(theta) * math.sin(phi)
      sample_points[i, 2] = math.cos(phi)
   return sample_points

if __name__ == '__main__':
   pass
