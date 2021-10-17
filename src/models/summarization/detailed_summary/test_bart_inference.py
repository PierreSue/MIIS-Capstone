# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: test_bart_inference.py
@time: 10/16/21
"""
import json
from time import time

from src.models.summarization.detailed_summary.bart_inference import detailed_summary


def run():
    input_path = '/home/jingrong/capstone/data/result-grammar_corrected.json'
    with open(input_path, 'r') as f_in:
        segments = json.load(f_in)

    transcripts = [seg['transcript'] for seg in segments]
    tic = time()
    summaries = detailed_summary(transcripts)
    toc = time()
    print(f"===== {toc - tic}s =====")

    assert len(summaries) == len(transcripts)
    for transcript, summary in zip(transcripts, summaries):
        print('[Transcript]', transcript, '\n')
        print('[Summary]', summary, '\n\n')


if __name__ == '__main__':
    run()
