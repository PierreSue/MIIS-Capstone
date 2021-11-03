# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: utils.py
@time: 10/14/21
"""
import json
import os
from datetime import datetime
from typing import List, Any

import nltk
nltk.download('punkt')

START_TIMESTAMP = datetime.strptime("00:00:00", "%H:%M:%S")
BI_THRESHOLD_OCR = 100


def segment_sentence(paragraph: str) -> List[str]:
    sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sen_tokenizer.tokenize(paragraph)
    return sentences


def word_tokenize(sentence: str) -> List[str]:
    return nltk.word_tokenize(sentence)


def add_key_value_into_jsonfile(output_file: str, key: str, value: Any, exist_ok: bool = False):
    if os.path.exists(output_file):
        with open(output_file, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = dict()
    if not exist_ok:
        assert key not in data, f"'{key}' already exists in {output_file}"
    data[key] = value
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
