# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: 1_segmentation.py
@time: 11/1/21
"""
import os

from src import SRC_PATH
from src.conf import DATA_DIR, VIDEO_ID


def run():
    output_path = f'{os.path.join(DATA_DIR, "results", VIDEO_ID)}.json'
    os.system(f'python {SRC_PATH}/models/segmentation/video_segmentation.py '
              f'--video-path {os.path.join(DATA_DIR, "video", VIDEO_ID)}.mp4 '
              f'--output-path {output_path}')
    print(f'Results have been saved to {output_path}')


if __name__ == '__main__':
    run()
