# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: 3_grammar_correction.py
@time: 10/15/21
"""
import json
import os

from tqdm import tqdm

from gingerit.gingerit import GingerIt

from src.utils import segment_sentence
from src.conf import DATA_DIR, VIDEO_ID


def grammar_correction(paragraph: str) -> str:
    sentences = segment_sentence(paragraph)
    parser = GingerIt()
    res_sentences = []
    for sen in tqdm(sentences, total=len(sentences), desc='Grammar Correction ..', ncols=80):
        res_sentences.append(parser.parse(sen)['result'])
    return ' '.join(res_sentences)


def run():
    result_file = os.path.join(DATA_DIR, "results", VIDEO_ID + '.json')
    with open(result_file, 'r') as json_file:
        data = json.load(json_file)
    for segment in data['segments']:
        segment['transcript_corrected'] = grammar_correction(segment['transcript'])
    with open(result_file, 'w') as f_out:
        json.dump(data, f_out, indent=4, ensure_ascii=False)
    print(f'Results have been saved to {result_file}')


if __name__ == '__main__':
    run()
