# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: grammar_correction.py
@time: 10/15/21
"""
import json

from tqdm import tqdm

from gingerit.gingerit import GingerIt

from src.preprocessing.utils import segment_sentence


def grammar_correction(paragraph: str) -> str:
    sentences = segment_sentence(paragraph)
    parser = GingerIt()
    res_sentences = []
    for sen in tqdm(sentences, total=len(sentences), desc='Grammar Correction ..', ncols=80):
        res_sentences.append(parser.parse(sen)['result'])
    return ' '.join(res_sentences)


def run():
    with open('/home/jingrong/capstone/data/segmentation/segmented_transcript.json', 'r') as f_in:
        boundaries_and_segments = json.load(f_in)
    for segment in boundaries_and_segments:
        segment['transcript-corrected'] = grammar_correction(segment['transcript'])
    with open('/home/jingrong/capstone/data/segmentation/segmented_transcript-grammar_corrected.json', 'w') as f_out:
        json.dump(boundaries_and_segments, f_out, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    run()
