# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: bart_inference.py
@time: 10/16/21
"""
from typing import List

import summertime.model as st_model


def detailed_summary(batch_paragraph: List[str]) -> List[str]:
    bart_model = st_model.BartModel(device='cuda')
    batch_summary = bart_model.summarize(batch_paragraph)
    return batch_summary
