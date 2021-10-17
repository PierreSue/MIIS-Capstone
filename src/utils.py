# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: utils.py
@time: 10/14/21
"""
from datetime import datetime
from typing import List

import nltk
nltk.download('punkt')


START_TIMESTAMP = datetime.strptime("00:00:00", "%H:%M:%S")


def segment_sentence(paragraph: str) -> List[str]:
    sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sen_tokenizer.tokenize(paragraph)
    return sentences
