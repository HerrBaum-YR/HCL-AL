from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

from md_clip3d.utils.kaiming_init import kaiming_weight_init
from md_clip3d.network.ClipText_BasicNet import TextEncoder
from md_clip3d.network.PubMedBERT_BasicNet import BasicNet


class ClipNet(torch.nn.Module):
   def __init__(self, in_channels, input_size, pretrained_model_dir, topK=None, loss_func=None, output_loss=False):
      super(ClipNet, self).__init__()

      self.text_encoder = TextEncoder(pretrained_model_dir)
      self.image_encoder = BasicNet(in_channels, input_size, num_features=512)  # depends on which image encoder you use

      self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
      self.loss_func = loss_func
      self.output_loss = output_loss

      self.topK = topK

   def encode_image(self, input_images):
      image_embeddings = self.image_encoder(input_images)
      return image_embeddings

   def encode_text(self, text):
      text_embeddings = self.text_encoder(text)  # [batch_size, n_ctx, d_model]
      return text_embeddings
   
   def forward(self, input_images, input_texts, other_texts):
      # input_images: [batch_size, channel, crop_size_x, crop_size_y, crop_size_z]
      # input_texts: [batch_size, max_length]
      # other_texts: [batch_size, num_texts-1, max_length]

      if not self.topK:
         self.topK = int(input_images.size(0) - 1)

      image_embeddings = self.encode_image(input_images)
      image_embeddings = image_embeddings / image_embeddings.norm(dim = 1, keepdim = True)

      text_embeddings = self.encode_text(input_texts)

      logit_scale = self.logit_scale.exp()

      logits_per_image = []
      for i in range(input_images.size(0)):
         image_embedding = image_embeddings[i, :].unsqueeze(0)  # [1, image_features]
         text_embedding = text_embeddings[i, :].unsqueeze(0)  # [1, text_features]
         other_text_embeddings = self.encode_text(other_texts[i, :, :])  # [num_texts-1, text_features]

         # normalize text embeddings
         all_text_embeddings = torch.cat([text_embedding, other_text_embeddings], dim=0)  # [num_texts, text_features]
         all_text_embeddings = all_text_embeddings / all_text_embeddings.norm(dim = 1, keepdim = True)

         # calcuate the similarity
         logit_per_image_all_texts = logit_scale * image_embedding @ all_text_embeddings.t()
         logit_per_image_input_text = logit_per_image_all_texts[:, 0]  # .unsqueeze(1)
         logit_per_image_other_texts = logit_per_image_all_texts[:, 1:]

         # select the topK highest similarity negative samples
         sorted_logit_per_image_other_texts, _ = torch.sort(logit_per_image_other_texts, dim=1, descending=True)
         logit_per_image_topK_texts = sorted_logit_per_image_other_texts[:, :self.topK]
   
         indices = sorted(list(range(self.topK)) + [min(i, self.topK - 1)])
         logit_per_image = logit_per_image_topK_texts[:, indices]
         logit_per_image[:, i] = logit_per_image_input_text
         
         logits_per_image.append(logit_per_image)

      logits_per_image = torch.cat(logits_per_image, dim=0)

      if self.output_loss:
         loss = self.loss_func(logits_per_image)
         return loss
      else:
         return logits_per_image
   
   def parameters_init(self):
      self.image_encoder.apply(kaiming_weight_init)
    
   @staticmethod
   def max_stride():
      return 16


if __name__ == "__main__":
   pass