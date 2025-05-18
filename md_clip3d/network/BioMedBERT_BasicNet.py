import os
import numpy as np

import torch
import torch.nn as nn
from transformers import BertModel

from md_clip3d.utils.kaiming_init import kaiming_weight_init
from md_clip3d.network.PubMedBERT_BasicNet import BasicNet


class ClipNet(torch.nn.Module):
   def __init__(self, in_channels, input_size, pretrained_model_dir, loss_func=None, output_loss=False):
      super(ClipNet, self).__init__()

      self.pretrained_model_dir = pretrained_model_dir
      
      self.image_encoder = BasicNet(in_channels, input_size, num_features=768)  # depends on which image encoder you use
      self.text_encoder = BertModel.from_pretrained(
         os.path.join(self.pretrained_model_dir, "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"))

      self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
      self.loss_func = loss_func
      self.output_loss = output_loss
   
   def encode_image(self, input_images):
      image_embeddings = self.image_encoder(input_images)
      return image_embeddings

   def encode_text(self, input_texts):
      text_embeddings = self.text_encoder(input_texts)['last_hidden_state']
      text_embeddings, _ = torch.max(text_embeddings, dim = 1)
      return text_embeddings

   def forward(self, input_images, input_texts):
      image_embeddings = self.encode_image(input_images)
      text_embeddings = self.encode_text(input_texts)

      image_embeddings = image_embeddings / image_embeddings.norm(dim = 1, keepdim = True)
      text_embeddings = text_embeddings / text_embeddings.norm(dim = 1, keepdim = True)

      logit_scale = self.logit_scale.exp()
      logits_per_image = logit_scale * image_embeddings @ text_embeddings.t()
      logits_per_text = logits_per_image.t()

      if self.output_loss:
         loss = self.loss_func(logits_per_image, logits_per_text)
         return loss
      else:
         return logits_per_image, logits_per_text
   
   def parameters_init(self):
      self.image_encoder.apply(kaiming_weight_init)
      state_dict_1 = torch.load(
         os.path.join(self.pretrained_model_dir, 'BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin'))
      state_dict_2 = self.state_dict()

      update = state_dict_2.copy()
      for key2, value2 in self.named_parameters():
         if not key2.startswith('text_encoder.'):
               continue
         suffix = key2.split('text_encoder.')[1]
         for key1, value1 in state_dict_1.items():
               if not key1.startswith('text.transformer.'):
                  continue
               _, after = key1.split('text.transformer.')[0], key1.split('text.transformer.')[1]
               if (suffix == after) and (value1.size() == value2.size()):
                  update[key2] = value1
      state_dict_2.update(update)
      self.load_state_dict(state_dict_2)
   
   @staticmethod
   def max_stride():
      return 16


if __name__ == "__main__":
   pass