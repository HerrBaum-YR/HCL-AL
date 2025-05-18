import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from md_clip3d.utils.kaiming_init import kaiming_weight_init
from md_clip3d.network.ClipText_BasicNet import TextEncoder


def parameters_init(net):
    net.apply(kaiming_weight_init)

class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=[96, 96, 96]):
        super(PatchEmbedding3D, self).__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.img_size = img_size
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)

        self.projection = nn.Conv3d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # [batch_size, emb_size, n_patches_d, n_patches_h, n_patches_w]
        x = x.flatten(2)  # [batch_size, emb_size, n_patches]
        x = x.transpose(1, 2)  # [batch_size, n_patches, emb_size]
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        assert self.head_dim * num_heads == emb_size, "Embedding size needs to be divisible by num_heads"

        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.fc = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        batch_size, n_tokens, emb_size = x.size()
        qkv = self.qkv(x)  # [batch_size, n_tokens, 3 * emb_size]
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.num_heads, self.head_dim)  # [batch_size, n_tokens, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, n_tokens, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(scores, dim=-1)

        x = (attn @ v).transpose(1, 2).contiguous().view(batch_size, n_tokens, self.emb_size)
        x = self.fc(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, ff_hidden_mult=4, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(emb_size)
        self.self_attention = MultiHeadSelfAttention(emb_size, num_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.layer_norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, ff_hidden_mult * emb_size),
            nn.GELU(),
            nn.Linear(ff_hidden_mult * emb_size, emb_size)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head self-attention
        attn_output = self.self_attention(self.layer_norm1(x))
        x = x + self.dropout1(attn_output)

        # feed-forward
        ff_output = self.feed_forward(self.layer_norm2(x))
        x = x + self.dropout2(ff_output)

        return x

class ViT(nn.Module):
    def __init__(self, in_channels, input_size, num_features, patch_size=16, emb_size=768, depth=12, num_heads=12, ff_hidden_mult=4, dropout=0.1):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding3D(in_channels, patch_size, emb_size, input_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.position_embedding = nn.Parameter(torch.randn(1, 1 + self.patch_embedding.n_patches, emb_size))

        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, ff_hidden_mult, dropout)
            for _ in range(depth)
        ])

        self.layer_norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, num_features)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embedding

        for layer in self.encoder:
            x = layer(x)

        x = self.layer_norm(x)
        cls_logits = self.fc(x[:, 0])
        return cls_logits

    @staticmethod
    def max_stride():
        return 16

class ClipNet(torch.nn.Module):
   def __init__(self, in_channels, input_size, pretrained_model_dir, loss_func=None, output_loss=False):
      super(ClipNet, self).__init__()

      self.text_encoder = TextEncoder(pretrained_model_dir)
      self.image_encoder = ViT(in_channels, input_size, num_features=512)  # depends on which image encoder you use

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