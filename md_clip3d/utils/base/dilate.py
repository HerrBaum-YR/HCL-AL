import torch
from md_clip3d.utils.clip_utils import bounding_box_voxel
import SimpleITK as sitk

def neighborhood3d(connectivity):
   """
   Get neighborhood of 6-connected or 26-connected ways

   Args:
      connectivity (int): connectivity 6 or 26 for an image3d object
   Return:
      list: neighborhood list (x, y, z)
   """
   neighbors = []
   if connectivity not in [6, 26]:
      raise ValueError("Invalid argument: connectivity has to be 6 or 26")

   if connectivity == 6:
      for x in [-1, 1]:
         neighbors.append((x, 0, 0))
      for y in [-1, 1]:
         neighbors.append((0, y, 0))
      for z in [-1, 1]:
         neighbors.append((0, 0, z))
   elif connectivity == 26:
      for z in range(-1, 2):
         for y in range(-1, 2):
               for x in range(-1, 2):
                  if x == 0 and y == 0 and z == 0:
                     continue
                  neighbors.append((x, y, z))
   return neighbors

def is_foreground(value, label):
   if label == 0:
      return value > 0
   else:
      return value == label

def sub_dilate(image, image_tmp, min_e, max_e, label, neighbors, check_edge=False):
   depth, height, width = image.shape
   for z in range(min_e[0], max_e[0] + 1):
      for y in range(min_e[1], max_e[1] + 1):
         for x in range(min_e[2], max_e[2] + 1):
               if is_foreground(image[z, y, x].item(), label):
                  continue
               for dz, dy, dx in neighbors:
                  z_n, y_n, x_n = z + dz, y + dy, x + dx
                  if check_edge:
                     if x_n < 0 or x_n >= width or y_n < 0 or y_n >= height or z_n < 0 or z_n >= depth:
                           continue
                  if 0 <= z_n < depth and 0 <= y_n < height and 0 <= x_n < width:
                     if is_foreground(image[z_n, y_n, x_n].item(), label):
                           image_tmp[z, y, x] = label
                           break

def imdilate(image: sitk.Image, iteration: int, label: int, connectivity: int):
   """
   Args:
      image (torch.Tensor): image tensor
      iteration (int): iteration of dilatation
      label (int): label to dilate
      neighbors (list): neighborhood list

   Return:
      torch.Tensor: dilated image tensor
   """
   neighbors = neighborhood3d(connectivity)

   min_label = label
   max_label = label
   if label == 0:
      min_label = 1
      max_label = torch.iinfo(torch.int32).max

   min_b, max_b = bounding_box_voxel(image, min_label, max_label)
   if not min_b or not max_b:
      return image
   
   min_b = list(min_b)
   max_b = list(max_b)

   image_npy = sitk.GetArrayFromImage(image)
   image_tmp = image_npy.copy()
   im_size = image_npy.shape
   for _ in range(iteration):
      for bi in range(3):
         min_b[bi] = max(min_b[bi] - 1, 0)
         max_b[bi] = min(max_b[bi] + 1, im_size[bi] - 1)
      min_org = min_b.copy()
      max_org = max_b.copy()
      for i in range(3):
         if min_b[i] == 0:
               max_e = max_org.copy()
               min_e = min_org.copy()
               max_e[i] = 0
               sub_dilate(image_npy, image_tmp, min_e, max_e, label, neighbors, True)
               min_b[i] = 1
         if max_b[i] == im_size[i] - 1:
               max_e = max_org.copy()
               min_e = min_org.copy()
               min_e[i] = im_size[i] - 1
               sub_dilate(image_npy, image_tmp, min_e, max_e, label, neighbors, True)
               max_b[i] = im_size[i] - 2

      sub_dilate(image_npy, image_tmp, min_b, max_b, label, neighbors, False)
      image_npy = image_tmp.copy()

   result = sitk.GetImageFromArray(image_npy)
   result.CopyInformation(image)
   return result

if __name__ == "__main__":
   impath = "/mnt/AI_Station/PET_CT_Fusion/project/data/volume/Guangfuyi/FDG/P48110_20220926/Body_CT_Reg.nii.gz"
   im_itk = sitk.ReadImage(impath)
   image_npy = sitk.GetArrayFromImage(im_itk)
   image_npy[image_npy != 0] = 0
   image_npy[100, 100, 100] = 1
   im_itk = sitk.GetImageFromArray(image_npy)
   dilated_image = imdilate(im_itk, 2, 1, 6)
   image_npy = sitk.GetArrayFromImage(dilated_image)
   print(image_npy[95:105, 95:105, 95:105])