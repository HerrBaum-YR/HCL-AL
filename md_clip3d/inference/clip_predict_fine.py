import os
import re
import numpy as np

from md_clip3d.utils.clip_fileio import load_json_as_dict
from md_clip3d.tokenizer.clip_tokenize import tokenize
from md_clip3d.inference.clip_pyimpl import net_classify, load_crop
from md_clip3d.inference.clip_inference_utils import remove_zero_prob_location, map_fine_to_coarse, query_location

def preprocessing(imname, box, cfg):
   inference_dict = load_json_as_dict(cfg.library.inference_json_path)
   words_to_texts_dict = load_json_as_dict(cfg.library.translate_json_path)
   texts_to_words_dict = {v: k for k, v in words_to_texts_dict.items()}

   gender, size = None, None
   # load gender info
   if cfg.fine.filter_by_gender:
      img_path = imname[0]
      img_json = load_json_as_dict(os.path.join(os.path.dirname(img_path), 'Body_CT.json'))
      gender = img_json['patient_sex']
      assert gender in ['F', 'M'], '[Fine]：Invalid gender info.'
   # load box size info
   if cfg.fine.filter_by_size:
      size = min([float(box['width']), float(box['height']), float(box['depth'])])
   

   coarse_words = [str(d) for d in re.split(',', box['coarse_labels'])]
   coarse_probs = [float(d) for d in re.split(',', box['coarse_probs'])]
   
   fine_words = []
   fine_to_coarse_words = []
   fine_to_coarse_probs = []
   for coarse_word, coarse_prob in zip(coarse_words, coarse_probs):
      for body in inference_dict:
         if coarse_word in inference_dict[body]:
            for fine_word in inference_dict[body][coarse_word]:
               condition = inference_dict[body][coarse_word][fine_word]
               # filter input texts by gender
               if gender and 'gender' in condition:
                  if gender != condition['gender']:
                     continue
               # filter input texts by box size
               if size and 'max_size' in condition:
                  if size > condition['max_size']:
                     continue
               if fine_word not in fine_words:
                  fine_words.append(fine_word)
                  fine_to_coarse_words.append(coarse_word)
                  fine_to_coarse_probs.append(coarse_prob)
      
   fine_texts = []
   for fine_word in fine_words:
      fine_texts.append(words_to_texts_dict[fine_word])
   return fine_texts, fine_to_coarse_words, fine_to_coarse_probs, texts_to_words_dict
               
def postprocessing(box, coarse_labels, coarse_probs, fine_labels, fine_probs, cfg):
   # fuse coarse and fine prob
   if max(fine_probs) < cfg.fine.fuse_coarse_thres:
      fused_fine_probs = [0.5 * coarse_probs[i] + 0.5 * fine_probs[i] for i in range(len(fine_probs))]
      fine_probs = fused_fine_probs
   
   # sort labels by probs
   sort_indices = list(np.argsort(-np.array(fine_probs)))
   coarse_labels = [coarse_labels[i] for i in sort_indices]
   fine_labels = [fine_labels[i] for i in sort_indices]
   fine_probs = [fine_probs[i] for i in sort_indices]

   fine_to_coarse_dict = dict()
   inference_dict = load_json_as_dict(cfg.library.inference_json_path)
   for body in inference_dict:
      for coarse_word in inference_dict[body]:
         for fine_word in inference_dict[body][coarse_word]:
            if fine_word not in fine_to_coarse_dict:
               fine_to_coarse_dict[fine_word] = []
            if coarse_word not in fine_to_coarse_dict[fine_word]:
                  fine_to_coarse_dict[fine_word].append(coarse_word)
   fine_to_coarse_labels = []
   for fine_label in fine_labels:
      fine_to_coarse_label = map_fine_to_coarse(fine_label, coarse_labels, fine_to_coarse_dict)
      fine_to_coarse_labels.append(fine_to_coarse_label)

   related_lists = [fine_to_coarse_labels]
   # filter out prediction localizations with zero probability
   fine_labels, fine_probs, related_lists = remove_zero_prob_location(fine_labels, fine_probs, related_lists)
   fine_to_coarse_labels = related_lists[0]

   box['fine_to_coarse_labels'] = ','.join(fine_to_coarse_labels)
   box['fine_labels'] = ','.join(fine_labels)
   box['fine_probs'] = ','.join([str(prob) for prob in fine_probs])
   coarse_and_fine_labels = ','.join([f"{fine_to_coarse_labels[i]}->{fine_labels[i]}" for i in range(len(fine_labels))][:5])
   rough_probs = ','.join(["{:.2f}".format(prob) for prob in fine_probs][:5])
   print(f"[Fine] Probs: {coarse_and_fine_labels}（{rough_probs}）")

   # for extremely large lesions, return coarse-grained prediction result
   if min([float(box['width']), float(box['height']), float(box['depth'])]) > 100:
      box['pred'] = box['coarse_labels'].split(',')[0]
   else:
      box['pred'] = query_location(fine_to_coarse_labels[0], fine_labels[0], inference_dict)

   print(f"Output: {box['pred']}")
   return box

def predict_fine(imname, box, images, model, net_name, cfg, target_label=0):
   
   bbox_info = []
   bbox_info.extend([float(box['x']), float(box['y']), float(box['z']), 
                     float(box['width']), float(box['height']), float(box['depth'])])
   
   # load images
   crop_images = load_crop(images, model, bbox_info, target_label)
   
   # load input texts
   fine_texts, coarse_words, coarse_probs, texts_to_words_dict = preprocessing(imname, box, cfg)

   # model inference
   num_texts = len(fine_texts)
   text_tokens = tokenize(fine_texts, net_name, cfg.general.pretrained_model_dir)
   unsort_probs, _ = net_classify(crop_images, text_tokens, model['clip'], num_texts, no_sort=True)
   unsort_probs = [float(unsort_probs[0][i]) for i in range(num_texts)]
   unsort_fine_words = [texts_to_words_dict[fine_texts[i]] for i in range(num_texts)]

   box = postprocessing(box, coarse_words, coarse_probs, unsort_fine_words, unsort_probs, cfg)
   return box
