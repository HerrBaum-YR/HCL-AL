import os
from transformers import BertTokenizer
from md_clip3d.tokenizer.simple_tokenizer import SimpleTokenizer


def tokenize(texts, net_name, model_dir):
    if not isinstance(texts, list):
        texts = [texts]

    if "PubMedBERT" in net_name:
        tokenizer = BertTokenizer.from_pretrained(
            os.path.join(model_dir, "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"))
        text_tokens = tokenizer.batch_encode_plus(
            [text for text in texts],
            max_length=77, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
    elif "BioMedBERT" in net_name:
        tokenizer = BertTokenizer.from_pretrained(
            os.path.join(model_dir, "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"))
        text_tokens = tokenizer.batch_encode_plus(
            [text for text in texts],
            max_length=77, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
    elif "ClipText" in net_name:
        tokenizer = SimpleTokenizer()
        text_tokens = tokenizer.tokenize(texts)
    else:
        raise ValueError(f"Invalid net name {net_name}")
    
    return text_tokens
