# encoding: utf-8
"""
@author: Jingrong Feng and Zhihao Wang
@contact: jingronf@andrew.cmu.edu and zhihaow2@andrew.cmu.edu
@version: 0.1
@file: transcript_preproc.py
@time: 10/5/21
"""
import math
from datetime import datetime
from typing import List

START_TIMESTAMP = datetime.strptime("00:00:00", "%H:%M:%S")


def timestamp_to_seconds(time_str: str):
    timestamp = datetime.strptime(time_str, "%H:%M:%S")
    return (timestamp - START_TIMESTAMP).total_seconds()


def merge_and_segment_transcript(segment_boundaries: List[str], caption_file: str):
    segment_boundaries_sec = [timestamp_to_seconds(t) for t in segment_boundaries]
    num_boundaries = len(segment_boundaries_sec)
    # print('*' * 30, 'Input Segmentation Boundaries', '*' * 30)
    # print('Timestamp\tSeconds')
    # for time_str, sec in zip(segment_boundaries, segment_boundaries_sec):
    #     print(f'{time_str}\t{sec}')
    segment_boundaries_sec.append(math.inf)

    transcript_segments = []
    cur_segment_end = segment_boundaries_sec[0]
    cur_segment_text = []
    with open(caption_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
    for row, line in enumerate(lines):
        row += 1
        line = line.strip().replace('um ', '')
        # blank line
        if line == '':
            assert row % 3 == 0 or row == len(lines) - 1
            continue
        # the last segment
        if len(transcript_segments) >= num_boundaries and row % 3 != 1:
            cur_segment_text.append(line)
        else:
            if row % 3 == 1:  # timestamp line
                caption_start_time = timestamp_to_seconds(line.split('.')[0])
                if caption_start_time >= cur_segment_end:
                    transcript_segments.append(' '.join(cur_segment_text))
                    cur_segment_end = segment_boundaries_sec[len(transcript_segments)]
                    cur_segment_text = [cur_segment_text[-1]]  # duplicate one caption on the boundary
            else:
                cur_segment_text.append(line)
    transcript_segments.append(' '.join(cur_segment_text))
    assert len(transcript_segments) == num_boundaries + 1
    return transcript_segments


if __name__ == '__main__':
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
        caption_file='/Users/FengJR/Downloads/CMU/Capstone_Project/data/captions_Human_Speech.sbv'
    )
    for segment in segments:
        print(segment)
        print('\n', '*' * 100, '\n')
