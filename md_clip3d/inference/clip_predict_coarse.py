from md_clip3d.utils.clip_fileio import load_json_as_dict
from md_clip3d.tokenizer.clip_tokenize import tokenize
from md_clip3d.inference.clip_pyimpl import net_classify, load_crop
from md_clip3d.inference.clip_inference_utils import remove_zero_prob_location

def preprocessing(box, cfg):
   # load json
   inference_dict = load_json_as_dict(cfg.library.inference_json_path)
   words_to_texts_dict = load_json_as_dict(cfg.library.translate_json_path)
   texts_to_words_dict = {v: k for k, v in words_to_texts_dict.items()}

   # load input texts
   words = []
   for body in inference_dict:
      for coarse_word in inference_dict[body]:
         if coarse_word not in words:
            words.append(coarse_word)

   texts = [words_to_texts_dict[word] for word in words]
   return texts, texts_to_words_dict

def postprocessing(box, labels, probs, cfg):
   # load config
   topk = cfg.coarse.topk

   # filter out prediction localizations with zero probability
   labels, probs, _ = remove_zero_prob_location(labels, probs)

   # Keep only the topK results
   if topk:
      labels = labels[:min(topk, len(labels))]
      probs = probs[:min(topk, len(probs))]
   
   box['coarse_labels'] = ','.join(labels)
   box['coarse_probs'] = ','.join([str(prob) for prob in probs])
   tmp_labels = ','.join(labels[:5])
   tmp_probs = ','.join(["{:.2f}".format(prob) for prob in probs[:5]])
   print(f"[Coarse] Probs: {tmp_labels}（{tmp_probs}）")
   return box

def predict_coarse(box, images, model, net_name, cfg, target_label=0):

   bbox_info = []
   bbox_info.extend([float(box['x']), float(box['y']), float(box['z']),
                     float(box['width']), float(box['height']), float(box['depth'])])
   
   # load images
   crop_images = load_crop(images, model, bbox_info, target_label)
   
   # filter input texts
   texts, texts_to_words_dict = preprocessing(box, cfg)
   
   # Model inference
   num_texts = len(texts)
   text_tokens = tokenize(texts, net_name, cfg.general.pretrained_model_dir)
   sorted_probs, sorted_labels = net_classify(crop_images, text_tokens, model['clip'], num_texts)
   sorted_probs = [float(sorted_probs[0][i]) for i in range(num_texts)]
   sorted_labels = [texts_to_words_dict[texts[sorted_labels[0].numpy()[i]]] for i in range(num_texts)]

   # filter output texts
   box = postprocessing(box, sorted_labels, sorted_probs, cfg)
   return box

