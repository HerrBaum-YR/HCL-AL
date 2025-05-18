from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

from md_clip3d.utils.kaiming_init import kaiming_weight_init
from md_clip3d.network.ClipText_BasicNet import TextEncoder
from md_clip3d.network.PubMedBERT_BasicNet import BasicNet


class ClipNet(torch.nn.Module):
   def __init__(self, in_channels, input_size, pretrained_model_dir,loss_func=None, output_loss=False):
      super(ClipNet, self).__init__()

      self.text_encoder = TextEncoder(pretrained_model_dir)
      self.text_fc = nn.Linear(512, 512)
      self.image_encoder = BasicNet(in_channels, input_size, num_features=512)  # depends on which image encoder you use

      self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
      self.loss_func = loss_func
      self.output_loss = output_loss

   def encode_image(self, input_images):
      image_embeddings = self.image_encoder(input_images)
      return image_embeddings

   def encode_text(self, text):
      text_embeddings = self.text_encoder(text)  # [batch_size, n_ctx, d_model]
      text_embeddings = self.text_fc(text_embeddings)
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
    
   @staticmethod
   def max_stride():
      return 16
   

if __name__ == "__main__":
    pass