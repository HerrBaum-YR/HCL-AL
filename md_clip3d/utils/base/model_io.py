import io
import torch
from md_clip3d.utils.base.crypto import Crypto


def load_pytorch_model(path):
   """
   :param path: model path
   :return: model params
   """
   with open(path, "rb") as fid:
      buffer = io.BytesIO(fid.read())
      buffer_value = buffer.getvalue()

      if buffer_value[0:9] == b"uAI_model":
         crypto_handle = Crypto()
         decrypt_buffer = io.BytesIO(crypto_handle.bytes_decrypt(buffer_value[128::]))
      else:
         decrypt_buffer = buffer
   params = torch.load(decrypt_buffer)
   return params


def save_pytorch_model(params, save_path, is_encrypt=True):
   """
   :param params: model params
   :param save_path: model save path
   :param is_encrypt: encrypt or not
   :return: None
   """
   if not is_encrypt:
      torch.save(params, save_path)
      return

   buffer = io.BytesIO()
   torch.save(params, buffer)
   tag = b"uAI_model"
   tag = tag + b'\0'*(128 - len(tag))

   crypto_handle = Crypto()
   encrypt_buffer = tag + crypto_handle.bytes_encrypt(buffer.getvalue())

   with open(save_path, "wb") as fid:
      fid.write(encrypt_buffer)

