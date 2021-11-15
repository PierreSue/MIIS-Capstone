# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: 5_key_concept_extraction.py
@time: 11/13/21
"""
import json
import os
from collections import Counter
from typing import List, Dict, Any, Optional, Set

from tqdm import tqdm

from src.conf import DATA_DIR, VIDEO_ID, KEYPHRASE_NGRAM_RANGE, NUM_KEY_CONCEPTS_PER_LECTURE, MAX_KEY_CONCEPTS_PER_SEGMENT
from src.models.key_concept_extraction.keyphrase_extraction import extract_keyphrases
from src.models.key_concept_extraction.keyword_extraction import extract_keywords
from src.models.key_concept_extraction.utils import get_nouns


def extract_key_concepts(text_list: List[str],
                         min_cnt_word: Optional[int] = None,
                         min_cnt_phrase: Optional[int] = None,
                         min_len_word: int = 4,
                         min_len_phrase: int = 7,
                         in_title_word: bool = False,
                         in_title_phrase: bool = False) -> Dict[str, Dict[str, Any]]:
    nouns_list = [get_nouns(text, lower=True) for text in text_list]
    keyword_to_info = extract_keywords(text_list, nouns_list, min_len_word=min_len_word)
    keyphrase_to_info = extract_keyphrases(text_list, nouns_list,
                                           min_len_phrase=min_len_phrase,
                                           ngram_range=KEYPHRASE_NGRAM_RANGE)

    # filter key concepts
    key_concepts = dict()
    for word, info in keyword_to_info.items():
        if in_title_word and not info['in_slide_title']:
            continue
        if min_cnt_word is not None and info['term_count_in_slides'] < min_cnt_word:
            continue
        key_concepts[word] = info
    for phrase, info in keyphrase_to_info.items():
        if in_title_phrase and not info['in_slide_title']:
            continue
        if min_cnt_phrase is not None and info['term_count_in_slides'] < min_cnt_phrase:
            continue
        key_concepts[phrase] = info

    # # remove 'speech' and 'processing' if 'speech processing' exists
    # for concept in list(key_concepts.keys()):
    #     # keyphrase only
    #     if ' ' not in concept:
    #         continue
    #     words = concept.split(' ')
    #     all_keywords = True
    #     for word in words:
    #         if word not in key_concepts:
    #             all_keywords = False
    #             break
    #     if all_keywords:
    #         for word in words:
    #             key_concepts.pop(word)

    # remove 'vowels' if 'vowel' exists
    for concept in list(key_concepts.keys()):
        if concept + 's' in key_concepts:
            key_concepts[concept]['term_count_in_slides'] += key_concepts.pop(concept + 's')['term_count_in_slides']
        elif concept + 'es' in key_concepts:
            key_concepts[concept]['term_count_in_slides'] += key_concepts.pop(concept + 'es')['term_count_in_slides']

    return key_concepts


def topk_concepts_for_lecture(key_concepts: Dict[str, Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    key_concepts_with_score = []
    for concept, info in key_concepts.items():
        key_concepts_with_score.append({
            'concept': concept,
            'score': info['term_count_in_slides'] + 4 * info['in_slide_title'] + 3 * (' ' in concept)
        })
    topk_concepts = sorted(key_concepts_with_score, key=lambda x: x['score'], reverse=True)[:k]
    return topk_concepts


def topk_concepts_for_segment(key_concepts: Dict[str, Dict[str, Any]],
                              transcript: str,
                              top_concepts_lecture: Set[str],
                              k: int,
                              min_cnt: int = 3) -> List[Dict[str, Any]]:
    transcript = transcript.lower()
    concept_counter = Counter({concept: transcript.count(concept) for concept in key_concepts
                               if concept in top_concepts_lecture})
    topk_concepts = [{
        'concept': concept,
        'term_count_in_transcript': cnt
    } for concept, cnt in concept_counter.most_common() if cnt >= min_cnt and key_concepts[concept]['in_slide_title']]
    return topk_concepts[:k]


def run():
    input_path = os.path.join(DATA_DIR, "results", VIDEO_ID + '.json')
    output_path = os.path.join(DATA_DIR, "results", VIDEO_ID + '.json')

    with open(input_path, 'r', encoding='utf8') as json_file:
        data = json.load(json_file)

    slides_text_list = [x['text'] for x in data['slides_ocr_texts']]
    key_concepts = extract_key_concepts(slides_text_list, min_len_word=4, min_len_phrase=7)
    print('key_concepts:', key_concepts.keys())
    data['key_concepts'] = key_concepts

    topk_concepts = topk_concepts_for_lecture(key_concepts=key_concepts, k=NUM_KEY_CONCEPTS_PER_LECTURE)
    print('topk_concepts:', topk_concepts)
    data['topk_concepts'] = topk_concepts

    for seg in tqdm(data['segments'], desc='Extract top k for each segment', total=len(data['segments']), ncols=80):
        seg['topk_concepts'] = topk_concepts_for_segment(key_concepts=key_concepts,
                                                         transcript=seg['transcript_corrected'],
                                                         top_concepts_lecture=set(x['concept'] for x in topk_concepts),
                                                         k=MAX_KEY_CONCEPTS_PER_SEGMENT,
                                                         min_cnt=2)

    with open(output_path, 'w', encoding='utf8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f'Results have been saved to {output_path}')


if __name__ == "__main__":
    run()
