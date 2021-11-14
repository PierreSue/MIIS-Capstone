# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: keyphrase_extraction.py
@time: 11/12/21
"""
import re
from collections import defaultdict
from typing import List, Dict, Any, Set

import nltk
from nltk.util import ngrams
from tqdm import tqdm

from src.models.key_concept_extraction.utils import query_in_dbpedia, get_nouns

INVALID_START = re.compile(r'^(and|of|as|the|a|an|to|is|\W) ')
# INVALID_END = re.compile(r' (and|of|as|the|a|an|to|is|\W)$')


def is_invalid_keyphrase(n_gram: str, nouns: Set[str], min_len: int) -> bool:
    if len(n_gram) < min_len:
        return True
    # Assume the last word must be NOUN
    if n_gram.split(' ')[-1] not in nouns:
        return True
    if re.search(INVALID_START, n_gram):
        return True
    return False


def extract_ngrams(sentence: str, ngram_range: List[int]) -> List[str]:
    """
    Function to generate n-grams from sentences.
    """
    word_list = nltk.word_tokenize(sentence)
    n_grams = []
    for num in ngram_range:
        n_grams.extend(ngrams(word_list, num))
    return [' '.join(grams) for grams in n_grams]


def extract_keyphrases(text_list: List[str],
                       nouns_list: List[Set[str]],
                       ngram_range: List[int],
                       min_len_phrase: int) -> Dict[str, Dict[str, Any]]:
    assert len(text_list) == len(nouns_list)

    keyphrase_to_response = dict()
    n_gram_cnt_total = defaultdict(int)
    n_gram_in_title = defaultdict(bool)
    n_gram_slide_indices = defaultdict(set)
    for slide_idx, (text, nouns) in tqdm(enumerate(zip(text_list, nouns_list)), desc=f'Extracting keyphrases',
                                         total=len(text_list), ncols=80):
        sentences = [sent.strip().lower() for sent in text.split('\n') if sent.strip()]
        for sent_idx, sent in enumerate(sentences):
            for n_gram in extract_ngrams(sent, ngram_range):
                if is_invalid_keyphrase(n_gram, nouns, min_len_phrase):
                    continue
                if n_gram not in n_gram_cnt_total:
                    results = query_in_dbpedia(n_gram, incl_desc=True)
                    if results:
                        keyphrase_to_response[n_gram] = results
                n_gram_slide_indices[n_gram].add(slide_idx)
                n_gram_cnt_total[n_gram] += 1
                if sent_idx == 0:
                    n_gram_in_title[n_gram] = True
        # print(keyphrase_to_response.keys())

    keyphrase_to_info = dict()
    for keyphrase, response in keyphrase_to_response.items():
        keyphrase_to_info[keyphrase] = {
            'term_count_in_slides': n_gram_cnt_total[keyphrase],
            'in_slide_title': n_gram_in_title[keyphrase],
            'slide_indices': sorted(n_gram_slide_indices[keyphrase]),
            'dbpedia_results': response
        }
    return keyphrase_to_info
