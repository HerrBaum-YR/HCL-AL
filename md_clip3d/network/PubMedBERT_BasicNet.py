import os
import numpy as np

import torch
import torch.nn as nn
from transformers import BertModel

from md_clip3d.utils.kaiming_init import kaiming_weight_init


class BasicBlock(nn.Module):
    """ downsample block of clip net """

    def __init__(self, in_channels):
        super(BasicBlock, self).__init__()
        out_channels = in_channels * 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.act(self.bn(self.conv(input)))
        return out

class InputBlock(nn.Module):
    """ input block of basic-net """

    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.act(self.bn(self.conv(input)))
        return out

class BasicNet(nn.Module):
    def __init__(self, in_channels, input_size, num_features):
        super().__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = BasicBlock(16)
        self.down_64 = BasicBlock(32)
        self.down_128 = BasicBlock(64)
        self.down_256 = BasicBlock(128)
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
        out = self.fc(out)
        return out

    @staticmethod
    def max_stride():
        return 16

class ClipNet(torch.nn.Module):
    def __init__(self, in_channels, input_size, pretrained_model_dir, loss_func=None, output_loss=False):
        super(ClipNet, self).__init__()
        
        self.image_encoder = BasicNet(in_channels, input_size, num_features=768)  # depends on which image encoder you use

        self.text_encoder = self.text_encoder = BertModel.from_pretrained(
            os.path.join(pretrained_model_dir, "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"))
        
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
    
    @staticmethod
    def max_stride():
        return 16
    

if __name__ == "__main__":
    pass
