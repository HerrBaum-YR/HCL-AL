from collections import OrderedDict
import os
import numpy as np

import torch
import torch.nn as nn

from md_clip3d.utils.kaiming_init import kaiming_weight_init
from md_clip3d.network.PubMedBERT_BasicNet import BasicNet

def convert_weights(model: nn.Module):
   """Convert applicable model parameters to fp16"""

   def _convert_weights_to_fp16(l):
      if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
         l.weight.data = l.weight.data.half()
         if l.bias is not None:
               l.bias.data = l.bias.data.half()

      if isinstance(l, nn.MultiheadAttention):
         for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
               tensor = getattr(l, attr)
               if tensor is not None:
                  tensor.data = tensor.data.half()

      for name in ["text_projection", "proj"]:
         if hasattr(l, name):
               attr = getattr(l, name)
               if attr is not None:
                  attr.data = attr.data.half()

   model.apply(_convert_weights_to_fp16)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        ret = super().forward(x.type(torch.float32))
        return ret


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    

class TextEncoder(nn.Module):
   def __init__(self, pretrained_model_dir):
      super().__init__()

      self.embed_dim = 512
      self.context_length = 77
      self.vocab_size = 49408
      self.transformer_width = 512
      self.transformer_heads = 8
      self.transformer_layers = 12

      self.transformer = Transformer(
            width=self.transformer_width,
            layers=self.transformer_layers,
            heads=self.transformer_heads,
            attn_mask=self.build_attention_mask()
        )
      self.vocab_size = self.vocab_size
      self.token_embedding = nn.Embedding(self.vocab_size, self.transformer_width)
      self.positional_embedding = nn.Parameter(torch.empty(self.context_length, self.transformer_width))
      self.ln_final = LayerNorm(self.transformer_width)
      self.text_projection = nn.Parameter(torch.empty(self.transformer_width, self.embed_dim))

      with open(os.path.join(pretrained_model_dir, "Clip-ViT-B-32/ViT-B-32.pt"), 'rb') as opened_file:
         model = torch.jit.load(opened_file)
      state_dict = model.state_dict()
      self.load_state_dict(state_dict, strict=False)
   
   def build_attention_mask(self):
      # lazily create causal attention mask, with full attention between the vision tokens
      # pytorch uses additive attention mask; fill with -inf
      mask = torch.empty(self.context_length, self.context_length)
      mask.fill_(float("-inf"))
      mask.triu_(1)  # zero out the lower diagonal
      return mask
   
   def forward(self, text):
      x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

      x = x + self.positional_embedding
      x = x.permute(1, 0, 2)  # NLD -> LND
      x = self.transformer(x)
      x = x.permute(1, 0, 2)  # lND -> NLD
      x = self.ln_final(x)

      # x.shape = [batch_size, n_ctx, transformer.width]
      # take features from the eot embedding (eot_token is the highest number in each sequence)
      x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
      return x


class ClipNet(torch.nn.Module):
   def __init__(self, in_channels, input_size, pretrained_model_dir, loss_func=None, output_loss=False):
      super(ClipNet, self).__init__()

      self.text_encoder = TextEncoder(pretrained_model_dir)
      self.image_encoder = BasicNet(in_channels, input_size, num_features=512)  # depends on which image encoder you use

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
   

if __name__ == "__main__":
   pass