from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from md_clip3d.utils.kaiming_init import kaiming_weight_init
from md_clip3d.utils.kaiming_init import kaiming_weight_init
from md_clip3d.network.ClipText_BasicNet import TextEncoder

class ResidualBlock3(nn.Module):
   def __init__(self, channels, ksize=3, pad=1, convs=2):
      super(ResidualBlock3, self).__init__()
      if convs <= 0 or ksize <= 0 or pad < 0:
         raise ValueError("Invalid parameters for ResidualBlock3D")
      
      self.convs = convs
      self.ops = nn.ModuleList()
      ksize3d = (ksize, ksize, ksize)
      stride3d = (1, 1, 1)
      pad3d = (pad, pad, pad)
      
      for i in range(convs):
         self.ops.append(
               nn.Sequential(
                  nn.Conv3d(channels, channels, kernel_size=ksize3d, stride=stride3d, padding=pad3d),
                  nn.BatchNorm3d(channels),
                  nn.ReLU() if i != convs - 1 else nn.Identity() 
               )
         )

   def forward(self, x):
      residual = x
      out = x
      for i, op in enumerate(self.ops):
         out = op(out)
      
      out = F.relu(out + residual)
      return out

class InputBlock(nn.Module):
   """ input block of vb-net """

   def __init__(self, in_channels, out_channels):
      super(InputBlock, self).__init__()
      self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
      self.bn = nn.BatchNorm3d(out_channels)
      self.act = nn.ReLU(inplace=True)

   def forward(self, input):
      out = self.act(self.bn(self.conv(input)))
      return out
   
class DownBlock(nn.Module):
   """ downsample block of v-net """
   def __init__(self, in_channels, num_convs, kernel_size=[2, 2, 2], stride=[2, 2, 2]):
      super(DownBlock, self).__init__()
      out_channels = in_channels * 2
      self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=1)
      self.down_bn = nn.BatchNorm3d(out_channels)
      self.down_act = nn.ReLU(inplace=True)
      self.rblock = ResidualBlock3(out_channels, 3, 1, num_convs)

   def forward(self, input):
      out = self.down_act(self.down_bn(self.down_conv(input)))
      out = self.rblock(out)
      return out

class ResidualNet(nn.Module):
   def __init__(self, in_channels, input_size, num_features):
      super(ResidualNet, self).__init__()
      self.in_block = InputBlock(in_channels, 16)
      self.down_32 = DownBlock(16, 1)
      self.down_64 = DownBlock(32, 2)
      self.down_128 = DownBlock(64, 3)
      self.down_256 = DownBlock(128, 3)

      self.fc = nn.Linear(
            256 * (input_size[0]//self.max_stride()) * 
            (input_size[1]//self.max_stride()) * 
            (input_size[2]//self.max_stride()), num_features)

   def forward(self, input):
      out16 = self.in_block(input)
      out32 = self.down_32(out16)
      out64 = self.down_64(out32)
      out128 = self.down_128(out64)
      out256 = self.down_256(out128)
      out = out256.view(out256.size(0), -1)
      # out = self.layer(out)
      out = self.fc(out)
      return out

   @staticmethod
   def max_stride():
      return 16

class ClipNet(torch.nn.Module):
   def __init__(self, in_channels, input_size, pretrained_model_dir, loss_func=None, output_loss=False):
      super(ClipNet, self).__init__()

      self.text_encoder = TextEncoder(pretrained_model_dir)
      self.image_encoder = ResidualNet(in_channels, input_size, num_features=512)  # depends on which image encoder you use

      self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
      self.loss_func = loss_func
      self.output_loss = output_loss

   def encode_image(self, input_images):
      image_embeddings = self.image_encoder(input_images)
      return image_embeddings

   def encode_text(self, text):
      text_embeddings = self.text_encoder(text)  # [batch_size, n_ctx, d_model]
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
   

def main():
   pass

if __name__ == "__main__":
   main()