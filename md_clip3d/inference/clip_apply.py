from __future__ import print_function
import os
import copy
import time
import argparse
import importlib
import numpy as np
import pandas as pd
import SimpleITK as sitk

from md_clip3d.utils.base.file_tools import load_module_from_disk
from md_clip3d.utils.clip_fileio import get_input_fieldname_from_csv, get_input_channels_from_fieldname, load_case_csv
from md_clip3d.utils.clip_utils import calc_lesion_bbox, count_labels, label_connected_component
from md_clip3d.inference.clip_pyimpl import classify_load_model
from md_clip3d.inference.clip_predict_coarse import predict_coarse
from md_clip3d.inference.clip_predict_fine import predict_fine


def run(config_file):
   # load configure file
   assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)
   cfg = load_module_from_disk(config_file)
   cfg = cfg.cfg

   # load input csv
   input_fieldname = get_input_fieldname_from_csv(cfg.general.input_path)
   input_channels = get_input_channels_from_fieldname(input_fieldname)
   case_imnames, case_list = load_case_csv(cfg.general.input_path, input_channels)

   if not os.path.isdir(cfg.general.output_folder):
      os.makedirs(cfg.general.output_folder)

   dataname = os.path.splitext(os.path.basename(cfg.general.input_path))[0]
   output_path = os.path.join(cfg.general.output_folder, f"{dataname}_{cfg.general.mode}_pred.csv")

   if cfg.coarse.run:
      print(f'Loading coarse model...')
      assert os.path.exists(cfg.coarse.model_dir), f'Cannot find model: {cfg.coarse.model_dir}'
      coarse_model, coarse_net_name = classify_load_model(cfg.coarse.model_dir, cfg.general.pretrained_model_dir, cfg.general.gpu_id)
   if cfg.fine.run:
      print(f'Loading fine model...')
      assert os.path.exists(cfg.fine.model_dir), f'Cannot find model: {cfg.fine.model_dir}'
      fine_model, fine_net_name = classify_load_model(cfg.fine.model_dir, cfg.general.pretrained_model_dir, cfg.general.gpu_id)

   num_cases = 0
   bboxes = []
   for imname, case in zip(case_imnames, case_list):
      num_cases += 1
      print('{}/{}: {}'.format(num_cases, len(case_imnames), imname[0]))

      # load images
      images = []
      for j in range(len(imname)):
         images.append(sitk.ReadImage(imname[j], outputPixelType=sitk.sitkFloat32))

      if cfg.general.mode == 'box':
         for box in case:
            if cfg.coarse.run:
               box = predict_coarse(box, images, coarse_model, coarse_net_name, cfg)
            if cfg.fine.run:
               box = predict_fine(imname, box, images, fine_model, fine_net_name, cfg)

            bboxes.append(box)
            print()

      elif cfg.general.mode == 'mask':
         lesion = images[coarse_model['clip']['lesion_idx'][0]]
         num_cpt, cpt_mask = label_connected_component(lesion, connectivity=26)
         for label in range(1, num_cpt + 1):
            box = dict()
            box['class'] = 'bbox'
            for key in case[0]:
               box[key] = case[0][key]

            center, size = calc_lesion_bbox(cpt_mask, label)
            box['x'], box['y'], box['z'] = center[0], center[1], center[2]
            box['width'], box['height'], box['depth'] = size[0], size[1], size[2] 

            if cfg.coarse.run:
               box = predict_coarse(box, images, coarse_model, coarse_net_name, cfg, label)
            if cfg.fine.run:
               box = predict_fine(imname, box, images, fine_model, fine_net_name, cfg)

            bboxes.append(box)
            print()
      else:
         raise ValueError("Invalid inference mode, only support for [box] and [mask].")

   if bboxes:
      df = pd.DataFrame(bboxes)
      df.to_csv(output_path, index=False)


def main():
   parser = argparse.ArgumentParser(description="UII CLIP3D Inference Engine")
   parser.add_argument('-i', '--input', type=str, nargs='?', help='clip inference config file')
   args = parser.parse_args()

   run(args.input)

if __name__ == '__main__':
   main()
