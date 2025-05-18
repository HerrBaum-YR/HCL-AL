import os
import csv
import argparse
import numpy as np

from md_clip3d.utils.clip_utils import box_iou

def match_gt_bbox(pred_bbox, gt_bboxes):
   pred_box_npy = np.array([float(pred_bbox['x']), float(pred_bbox['y']), float(pred_bbox['z']), float(pred_bbox['width']), float(pred_bbox['height']), float(pred_bbox['depth'])])  
   iou_max = 0.5
   gt_locs = []
   for gt in gt_bboxes:
      gt_box_npy = np.array([float(gt['x']), float(gt['y']), float(gt['z']), float(gt['width']), float(gt['height']), float(gt['depth'])])
      iou = box_iou(pred_box_npy, gt_box_npy)
      if iou > iou_max:
         gt_locs = gt['GT'].split(',')
         iou_max = iou
   return gt_locs

def clip_evaluate(result_csv, gt_csv):
   # load gt csv
   gt_dict = dict()
   with open(gt_csv, 'r', encoding='utf-8-sig') as f:
      reader = csv.DictReader(f)
      for line in reader:
         casename = os.path.basename(os.path.dirname(line['image_path']))
         if casename not in gt_dict:
            gt_dict[casename] = []
         gt_dict[casename].append(line)
            
   # load prediction
   pred_dict = {}
   with open(result_csv, 'r', encoding='utf-8-sig') as fp:
      reader = csv.DictReader(fp)
      for line in reader:
         casename = os.path.basename(os.path.dirname(line['image_path']))
         if casename not in pred_dict:
            pred_dict[casename] = []
         pred_dict[casename].append(line)
   
   total_lesion_num, eval_lesion_num, correct_num = 0, 0, 0
   for casename in pred_dict:
      for line in pred_dict[casename]:
         total_lesion_num += 1

         pred_loc = line['pred']
         
         # match ground truth by lesion coordinate
         gt_locs = match_gt_bbox(line, gt_dict[casename])
         if not gt_locs:
            # print(f"Cannot match the valid ground truth.")
            continue

         # evaluate
         eval_lesion_num += 1
         if pred_loc in gt_locs:
            correct_num += 1

   print(f"Total lesion num：{total_lesion_num}")
   print(f"Evaluated lesion num：{eval_lesion_num}")
   print(f"Accuarcy (Top 1)：{correct_num / eval_lesion_num}")
   print()

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="UII CLIP3D Inference Engine")
   parser.add_argument('-i', '--input', type=str, nargs='?', help='clip inference input file')
   parser.add_argument('-g', '--gt', type=str, nargs='?', help='ground truth file')
   args = parser.parse_args()
   clip_evaluate(args.input, args.gt)