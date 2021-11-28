# encoding: utf-8
"""
@author: Zhihao Wang
@contact: zhihaow2@andrew.cmu.edu
@version: 0.1
@file: abstractive_sum.py
@time: 9/29/21

Return:
List[str]: timestamps of segment boundaries in HH:MM:SS format
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import gdown
from src.conf import MODEL_DIR
from zipfile import ZipFile

def brief_summary(srces):
    model_name = 'google/pegasus-xsum'
    device = torch.device('cuda')
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    train_path = "./brief_model/"
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(train_path + "train_model/pytorch_model.bin"):
        print("no model, downloading model..........")
        url = 'https://drive.google.com/uc?id=1JHZsdDXu52Ey9uHV8Qpkj8SC7aNDZXpv&export=download'
        output = 'train_model.zip'
        gdown.download(url, output, quiet=False)
        with ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(train_path)
    model = PegasusForConditionalGeneration.from_pretrained(train_path+"train_model/").to(device)
    res = []
    for text in srces:
        src = [text]
        batch = tokenizer(src, truncation=True, padding='longest', return_tensors="pt").to(device)
        translated = model.generate(**batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        tmp_text = tgt_text[0]
        tmp_text = tmp_text.replace(" /S>","").replace("/S>","")
        tmp_tgt = tmp_text.split("S> ")
        if len(tmp_tgt) > 1:
            tgt = tmp_tgt[1]
        else:
            tgt = tmp_tgt[0]
        tgt = tgt.split(".")[0] + "."
        res.append(tgt)
    return res

if __name__ == "__main__":
    srces = ['Test transcript 0','Test transcript']
    brief_summary(srces)