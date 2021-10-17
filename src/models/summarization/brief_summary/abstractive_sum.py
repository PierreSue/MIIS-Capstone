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


def brief_summary(srces):
    model_name = 'google/pegasus-xsum'
    device = torch.device('cuda')
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    res = []
    for text in srces:
        src = [text]
        batch = tokenizer(src, truncation=True, padding='longest', return_tensors="pt").to(device)
        translated = model.generate(**batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        res.append(tgt_text[0])
    return res
