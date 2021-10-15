# encoding: utf-8
"""
@author: Jingrong Feng and Zhihao Wang
@contact: jingronf@andrew.cmu.edu and zhihaow2@andrew.cmu.edu
@version: 0.1
@file: transcript_preproc_on_youtube_asr.py
@time: 10/5/21
"""
import json
import math
import os
from datetime import datetime
from typing import List

from tqdm import tqdm

from src.models.punctuation_restoration import PUNCTUATION_RESTORATION_PATH
from src.preprocessing.utils import START_TIMESTAMP, segment_sentence


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
    for row, line in enumerate(lines):
        row += 1
        line = line.strip()
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
                    transcript_segments.append(' '.join(cur_segment_text).replace('um ', ''))
                    cur_segment_end = segment_boundaries_sec[len(transcript_segments)]
                    cur_segment_text = [cur_segment_text[-1]]  # duplicate one caption on the boundary
            else:
                cur_segment_text.append(line)
    transcript_segments.append(' '.join(cur_segment_text).replace('um ', ''))
    assert len(transcript_segments) == num_boundaries + 1
    return transcript_segments


def restore_punctuation_and_capitalization(transcript_segments: List[str]) -> List[str]:
    base_dir = PUNCTUATION_RESTORATION_PATH
    restored_transcript_segments = []
    in_file = os.path.join(base_dir, 'data/tmp.txt')
    out_file = os.path.join(base_dir, 'data/tmp_out.txt')
    for segment in tqdm(transcript_segments, desc='Punctuation and Capitalization Restoration ..', ncols=80):
        with open(in_file, 'w') as f_out:
            f_out.write(segment)
        os.system(f'python {base_dir}/src/inference.py --pretrained-model=roberta-large '
                  f'--weight-path={base_dir}/weights/roberta-large-en.pt '
                  f'--language=en --in-file={in_file} --out-file={out_file}')
        with open(out_file, 'r') as f_in:
            punctuated_segment = f_in.read()
        sentences = segment_sentence(punctuated_segment)
        capitalized_segment = ' '.join([sen.capitalize() for sen in sentences])
        restored_transcript_segments.append(capitalized_segment)
    os.remove(in_file)
    os.remove(out_file)
    return restored_transcript_segments


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
        caption_file='/home/jingrong/capstone/data/segmentation/captions_Human_Speech.sbv'
    )
    print('\n\n', '*' * 100, 'Before Punctuation and Capitalization Restoration', '*' * 100, '\n\n')
    for segment in segments:
        print(segment)
        print('\n', '*' * 100, '\n')

    restored_transcript_segments = restore_punctuation_and_capitalization(segments)
    print('\n\n', '*' * 100, 'After Punctuation and Capitalization Restoration', '*' * 100, '\n\n')
    for segment in restored_transcript_segments:
        print(segment)
        print('\n', '*' * 100, '\n')

    boundaries = ["00:00:00"] + boundaries
    assert len(boundaries) == len(restored_transcript_segments)
    boundaries_and_segments = [{
        "start_timestamp": start_s,
        "transcript": segment
    } for start_s, segment in zip(boundaries, restored_transcript_segments)]
    with open('/home/jingrong/capstone/data/segmentation/segmented_transcript.json', 'w') as json_file:
        json.dump(boundaries_and_segments, json_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    run()
