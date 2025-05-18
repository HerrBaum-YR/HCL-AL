from Crypto.Cipher import AES
import base64


class Crypto(object):
   """Crypto provide bytes encrypt and decrypt function which mixes AES and base64."""
   def __init__(self, key=None):
      """
      :param key: password
      """
      if key is None:
         key = b"*c!q9Kj*k?2>+5@p"
      assert len(key) == 16
      self.key = key
      self.mode = AES.MODE_CFB

   def bytes_encrypt(self, plain_text):
      """
      :param plain_text:
      :return: cipher_text(bytes)
      """
      assert isinstance(plain_text, bytes)

      length = 16
      plain_text = plain_text + b'\1'
      count = len(plain_text)
      add = length - (count % length)
      plain_text = plain_text + (b'\0' * add)

      aes_handle = AES.new(self.key, self.mode, self.key)
      cipher_text = base64.b64encode(aes_handle.encrypt(plain_text))

      return cipher_text

   def bytes_decrypt(self, cipher_text):
      """
      :param cipher_text:
      :return: plaintext(bytes)
      """
      assert isinstance(cipher_text, bytes)

      aes_handle = AES.new(self.key, self.mode, self.key)
      plain_text = aes_handle.decrypt(base64.b64decode(cipher_text))
      
      return plain_text.rstrip(b'\0')[0:-1]


def main():
   handle = Crypto()
   cipher_text = handle.bytes_encrypt(b"uAI_model")
   print(cipher_text)
   plaintext = handle.bytes_decrypt(cipher_text)
   print(plaintext)


if __name__ == "__main__":
   main()
