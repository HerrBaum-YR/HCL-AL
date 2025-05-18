import re
import random
from torch.utils.data.sampler import BatchSampler

from md_clip3d.utils.clip_fileio import load_json_as_dict


class CoarsePositionBatchSampler(BatchSampler):
   def __init__(self, dataset, batchsize, epochs, location_json, region_sample_frequency, 
                target_header, region_num_per_batch, word_iter_num_per_epoch):

      self.dataset = dataset

      self.batchsize = batchsize
      self.epochs = epochs
      self.target_header = target_header
      
      # load json
      if location_json.endswith('json') :
         words_dict = load_json_as_dict(location_json)
      else:
         raise ValueError('location_json must be a json file')
      
      self.words = [list(words_dict[i].keys()) for i in words_dict]
      self.sample_word_idx_dict = self.stat_words_and_images()
      assert len(self.sample_word_idx_dict) >= self.batchsize, f'Number of text categories in the training data ({len(self.sample_word_idx_dict)}) < batchsize ({self.batchsize})'

      if region_sample_frequency:
         assert len(region_sample_frequency) == len(self.words)
      self.region_sample_frequency = region_sample_frequency

      # Each batch samples <batch_size> words from <region_num_per_batch> regions
      self.region_num_per_batch = region_num_per_batch

      # Each epoch consists of iterating over all words <word_iter_num_per_epoch> times
      self.word_iter_num_per_epoch = word_iter_num_per_epoch
      self.batch_per_epoch = self.word_iter_num_per_epoch * len(list(self.sample_word_idx_dict.keys()))

      words_no_sample = []
      for region in words_dict:
         for coarse in words_dict[region]:
            if coarse not in self.sample_word_idx_dict:
               words_no_sample.append(coarse)
      if len(words_no_sample):
         print(f"Samples at the following locations cannot be found: {','.join(words_no_sample)}")

   def stat_words_and_images(self):
      sample_word_idx_dict = dict()
      # Iterate through all lesions
      for box_idx, box_info in enumerate(self.dataset.other_info_list):
         cl = box_info[self.target_header]
         valid = False
         for ws in self.words:
            if cl in ws:
                valid = True
         if valid:
             if cl not in sample_word_idx_dict:
                sample_word_idx_dict[cl] = []
             sample_word_idx_dict[cl].append(box_idx)
      return sample_word_idx_dict
   
   def __iter__(self):
      for _ in range(len(self)):
         w_to_be_selected = []
         # Sample <region_num_per_batch> regions according to <region_sample_frequency>
         while (len(w_to_be_selected) < self.batchsize):
            p_idx_selected = set(random.choices(range(len(self.words)), weights=self.region_sample_frequency, k=self.region_num_per_batch))
            for p_idx in p_idx_selected:
               for w in self.words[p_idx]:
                  if w not in w_to_be_selected and w in self.sample_word_idx_dict:
                     w_to_be_selected.append(w)

         # sample <batchsize> words from these regions
         w_selected = random.sample(w_to_be_selected, k=self.batchsize)

         # for each word, sample one corresponding lesion ID
         indices = [random.choice(self.sample_word_idx_dict[w]) for w in w_selected]
         yield indices

   def __len__(self):
      return self.batch_per_epoch // self.batchsize * self.epochs

class FinePositionBatchSampler(BatchSampler):
   def __init__(self, dataset, batchsize, epochs, location_json, region_sample_frequency, 
                coarse_header, fine_header, related_header, related_rate, case_per_epoch):
      self.dataset = dataset

      self.batchsize = batchsize
      self.epochs = epochs
      self.coarse_header = coarse_header
      self.fine_header = fine_header
      self.related_header = related_header
      self.related_rate = related_rate
      self.case_per_epoch = case_per_epoch

      self.batch_per_epoch = self.case_per_epoch * self.batchsize
      self.related_word_num = int(self.batchsize * self.related_rate)
      
      # load json
      if location_json.endswith('json') :
         self.words_dict = load_json_as_dict(location_json)
      else:
         raise ValueError('location_json must be a json file')

      self.c_sample_word_idx_dict, self.cf_sample_word_idx_dict = self.stat_words_and_images()
      
      self.coarse_words = [list(self.words_dict[p].keys()) for p in self.words_dict]
      if region_sample_frequency:
         assert len(region_sample_frequency) == len(self.coarse_words)
      self.region_sample_frequency = region_sample_frequency

   def stat_words_and_images(self):
      c_sample_word_idx_dict, cf_sample_word_idx_dict = dict(), dict()
      # Iterate through all lesions
      for box_idx, box_info in enumerate(self.dataset.other_info_list):
         cl = box_info[self.coarse_header]
         fl = box_info[self.fine_header]
         if cl not in cf_sample_word_idx_dict:
            c_sample_word_idx_dict[cl] = []
            cf_sample_word_idx_dict[cl] = dict()
         if fl not in cf_sample_word_idx_dict[cl]:
            cf_sample_word_idx_dict[cl][fl] = []
         c_sample_word_idx_dict[cl].append(box_idx)
         cf_sample_word_idx_dict[cl][fl].append(box_idx)
      return c_sample_word_idx_dict, cf_sample_word_idx_dict
         
   def __iter__(self):
      for _ in range(len(self)):
         # Sample one major region according to <region_sample_frequency>
         p_idx_selected = random.choices(range(len(self.coarse_words)), weights=self.region_sample_frequency)[0]
         # from the selected major region, sample one location
         c_selected = random.choice([c for c in self.coarse_words[p_idx_selected] if c in self.c_sample_word_idx_dict])
         # from this location, sample one lesion
         case_idx_selected = random.choice(self.c_sample_word_idx_dict[c_selected])

         # Retrieve the ground truth coarse localization and model prediction for this lesion
         other_info = self.dataset.other_info_list[case_idx_selected]
         coarse_words = [other_info[self.coarse_header]]
         coarse_preds = [d for d in re.split(',', other_info[self.related_header])]
         for coarse_pred in coarse_preds:
            if coarse_pred not in coarse_words:
               coarse_words.append(coarse_pred)

         cf_selected = []
         for coarse_word in coarse_words:
            for fine_word in self.cf_sample_word_idx_dict[coarse_word]:
               cf_selected.append((coarse_word, fine_word))

         # Select first <related_word_num words>, the rest are sampled from fine localization vocabulary
         cf_selected = cf_selected[:self.related_word_num]
         while len(cf_selected) < self.batchsize:
            # Sample 1 major region according to <region_sample_frequency>
            p_idx_selected = random.choices(range(len(self.coarse_words)), weights=self.region_sample_frequency)[0]
            # Sample one coarse localization from the selected major region
            c_selected = random.choice([c for c in self.coarse_words[p_idx_selected] if c in self.c_sample_word_idx_dict])
            # Sample one fine localization from the selected coarse localization
            f_selected = random.choice(list(self.cf_sample_word_idx_dict[c_selected].keys()))
            if (c_selected, f_selected) not in cf_selected:
               cf_selected.append((c_selected, f_selected))

         # for each word, sample one corresponding lesion ID
         indices = []
         for (c, f) in cf_selected:
            index = random.choice(self.cf_sample_word_idx_dict[c][f])
            indices.append(index)
         yield indices

   def __len__(self):
      return self.case_per_epoch * self.epochs
