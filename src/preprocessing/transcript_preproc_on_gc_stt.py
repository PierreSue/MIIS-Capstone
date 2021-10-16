# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: transcript_preproc_on_gc_stt.py
@time: 10/15/21
"""
import math
from datetime import datetime
from typing import List

from src.preprocessing.utils import START_TIMESTAMP


def timestamp_to_seconds(time_str: str):
    timestamp = datetime.strptime(time_str, "%H:%M:%S")
    return (timestamp - START_TIMESTAMP).total_seconds()


def merge_and_segment_transcript(segment_boundaries: List[str], caption_file: str):
    segment_boundaries_sec = [timestamp_to_seconds(t) for t in segment_boundaries]
    num_boundaries = len(segment_boundaries_sec)
    segment_boundaries_sec.append(math.inf)

    transcript_segments = []
    cur_segment_end = segment_boundaries_sec[0]
    cur_segment_text = []
    with open(caption_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
    for line in lines:
        # no blank line
        assert len(line) > 0
        start_s, end_s, word = line.strip().split(' | ')
        # the last segment
        if len(transcript_segments) >= num_boundaries:
            cur_segment_text.append(word)
        else:
            caption_start_time = timestamp_to_seconds(start_s.split('.')[0])
            if caption_start_time >= cur_segment_end:
                # look backward to previous 9 words and segment earlier if there is an EOS punctuation.
                segment_earlier = False
                if cur_segment_text[-1][-1] not in '.?!':
                    for step in range(2, min(11, len(cur_segment_text) + 1)):
                        if cur_segment_text[-step][-1] in '.?!':
                            transcript_segments.append(' '.join(cur_segment_text[: -step + 1]).replace('um ', ''))
                            cur_segment_end = segment_boundaries_sec[len(transcript_segments)]
                            cur_segment_text = cur_segment_text[-step + 1:]
                            segment_earlier = True
                            break
                if not segment_earlier:
                    transcript_segments.append(' '.join(cur_segment_text).replace('um ', ''))
                    cur_segment_end = segment_boundaries_sec[len(transcript_segments)]
                    cur_segment_text = []
            cur_segment_text.append(word)
    transcript_segments.append(' '.join(cur_segment_text).replace('um ', ''))
    assert len(transcript_segments) == num_boundaries + 1
    return transcript_segments


def run():
    boundaries = [
        "00:01:16",
        "00:03:42",
        "00:12:16",
        "00:15:08",
        "00:16:52",
        "00:18:36",
        "00:21:36",
        "00:23:56",
        "00:26:10",
        "00:32:56",
        "00:36:30",
        "00:37:42",
        "00:38:42",
        "00:42:46",
        "00:43:50",
        "00:49:34",
        "00:54:30",
        "00:55:16",
        "00:59:38",
        "01:03:16",
        "01:06:00",
        "01:07:44"
    ]
    segments = merge_and_segment_transcript(
        segment_boundaries=boundaries,
        caption_file='/home/jingrong/capstone/data/gc_stt/2020_09_02_11_492_02_Human_Speech_transcript.txt'
    )
    for segment in segments:
        print(segment)
        print('\n', '*' * 100, '\n')


if __name__ == '__main__':
    run()
