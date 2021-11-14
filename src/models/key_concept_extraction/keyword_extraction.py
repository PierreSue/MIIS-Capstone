# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: keyword_extraction.py
@time: 11/13/21
"""
from collections import defaultdict
from typing import List, Dict, Any, Set

import nltk
from tqdm import tqdm

from src.models.key_concept_extraction.utils import query_in_dbpedia


def extract_keywords(text_list: List[str],
                     nouns_list: List[Set[str]],
                     min_len_word: int) -> Dict[str, Dict[str, Any]]:
    assert len(text_list) == len(nouns_list)

    keyword_to_response = dict()
    word_cnt_total = defaultdict(int)
    word_in_title = defaultdict(bool)
    word_slide_indices = defaultdict(set)
    for slide_idx, (text, nouns) in tqdm(enumerate(zip(text_list, nouns_list)), desc=f'Extracting keywords',
                                         total=len(text_list), ncols=80):
        sentences = [sent.strip().lower() for sent in text.split('\n') if sent.strip()]
        for sent_idx, sent in enumerate(sentences):
            for word in nltk.word_tokenize(sent):
                if len(word) < min_len_word or word not in nouns:
                    continue
                if word not in word_cnt_total:
                    results = query_in_dbpedia(word, incl_desc=True)
                    if results:
                        keyword_to_response[word] = results
                word_slide_indices[word].add(slide_idx)
                word_cnt_total[word] += 1
                if sent_idx == 0:
                    word_in_title[word] = True
        # print(keyword_to_response.keys())
    keyword_to_info = dict()
    for keyphrase, response in keyword_to_response.items():
        keyword_to_info[keyphrase] = {
            'term_count_in_slides': word_cnt_total[keyphrase],
            'in_slide_title': word_in_title[keyphrase],
            'slide_indices': sorted(word_slide_indices[keyphrase]),
            'dbpedia_results': response
        }
    return keyword_to_info
