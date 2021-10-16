# encoding: utf-8
"""
@author: Jingrong Feng and Pierre Su
@contact: jingronf@andrew.cmu.edu and fengguas@andrew.cmu.edu
@version: 0.1
@file: video_segmentation.py
@time: 9/29/21

Command Arguments
--video-path: The input video path
--interval: The interval between each image. (An image/Nsec)
--threshold-pixel: The threshold of pixel differences (1st stage)
--boundaries: The boundaries of slides: 'x_topleft, y_topleft, x_bottomright, y_bottomright'
--edit-distance-threshold: The maximum (Levenshtein_distance(s_{i}, s_{i+1}) / s_{i+1}) between two slides s_{i} and s_{i+1} within a segment (2nd stage)
--min-interval: The minimum time interval of a segment (in seconds)
--num-frame-forward: The number of frames we look forward to determine the boundaries (2nd stage)
--bi-threshold: The threshold for image binarization (0, 255), 0->black and 1->white (2nd stage)

Return:
List[str]: timestamps of segment boundaries in HH:MM:SS format
"""
import argparse
from time import strftime, gmtime
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from src.models.segmentation.utils import is_segment_boundary, OCRTextExtractor


def video_segmentation(opt) -> List[str]:
    assert opt.num_frame_forward >= 3, 'At least 3 frames forward'
    
    filename = opt.video_path

    # Read the given video
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_FPS, 1)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) * opt.interval
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print('filename {}:'.format(filename))
    print('\tfps:{}'.format(fps))
    print('\tframeCount:{}'.format(frame_count))
    print('\tframe Dimension:{}x{}'.format(frame_width, frame_height))

    images = np.zeros((frame_count // fps + 1, frame_height, frame_width, 3), np.dtype('uint8'))
    binaryimages = np.zeros((frame_count // fps + 1, frame_height, frame_width), np.dtype('uint8'))

    # Capture images from the video
    fc, ret = 0, True
    pbar = tqdm(total=frame_count // fps + 1, desc='Sample images from video', ncols=80)
    while (fc < frame_count and ret):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fc)
        ret, images[fc // fps] = cap.read()

        frame = cv2.cvtColor(images[fc // fps], cv2.COLOR_BGR2GRAY)
        ret, binaryimages[fc // fps] = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

        fc += fps
        pbar.update(1)
    cap.release()
    pbar.close()

    print("Finish reading the video:")
    print('\tshape:{}'.format(images.shape))

    # Crop the images into slides by the given boundaries
    boundaries = [int(boundary) for boundary in opt.boundaries.split(',')]
    boundaries = [(boundaries[0], boundaries[1]), (boundaries[2], boundaries[3])]
    slides = binaryimages[:, boundaries[0][0]:boundaries[1][0], boundaries[0][1]:boundaries[1][1]]

    # Calculate the pixel differences
    diff_areas, cutpoints = [], []
    for image_id, (imageA, imageB) in enumerate(zip(slides[:-1], slides[1:])):
        m, n = imageA.shape[0], imageA.shape[1]
        diff_map = np.abs(imageA - imageB)
        diff_map[diff_map > 0] = 1

        if np.sum(diff_map) == 0:
            diff_areas.append(0)
        else:
            diff_areas.append(np.sum(diff_map) / (m * n))
            if np.sum(diff_map) / (m * n) > opt.threshold_pixel:
                cutpoints.append((image_id + 1) * opt.interval)

    for i, cutpoint in enumerate(cutpoints):
        cutpoints[i] = (cutpoint // opt.interval, cutpoint)

    print("Cutpoints at the first stage:")
    for idx, cutpoint in enumerate(cutpoints):
        print(idx + 1, '\t', strftime("%H:%M:%S", gmtime(cutpoint[1])))

    # images: the images stored using numpy -> (num_frames, frameHeight, frameWidth, 3)
    # cutpoints: the list of the cutpoints(tuples) -> [(image_id, timestamp_in_seconds), (), ...]

    ocr_extractor = OCRTextExtractor()
    valid_boundaries = []
    last_boundary = 0
    for idx, (image_id, second) in tqdm(enumerate(cutpoints), total=len(cutpoints), desc='Second stage', ncols=80):
        # the first and last one frames cannot be boundaries
        if image_id < 3 or image_id == len(images) - 1:
            continue
        # OCR on a window of [prev_frames] + [current_frame, next_frame]
        num_prev_frame = image_id - max(image_id - opt.num_frame_forward, 0)
        batch_second = list(range(second - num_prev_frame * opt.interval, second + 2 * opt.interval, opt.interval))
        batch_text = ocr_extractor.batch_ocr(batch_image=images[image_id - num_prev_frame: image_id + 2],
                                             batch_second=batch_second,
                                             bi_threshold=opt.bi_threshold)
        assert len(batch_text) == len(batch_second)
        batch_second_text = list(zip(batch_second, batch_text))
        # determine if the current frame is a boundary
        if is_segment_boundary(batch_second_text, cur_idx=num_prev_frame, last_boundary=last_boundary,
                               threshold_ratio=opt.edit_distance_threshold, min_interval=opt.min_interval):
            valid_boundaries.append(strftime("%H:%M:%S", gmtime(second)))
            last_boundary = second

    return valid_boundaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, default='../Data/11492_02_Human_Speech.mp4',
                        help='The input video path')
    parser.add_argument('--interval', type=int, default=2,
                        help='The interval between each image in seconds. (An image/N sec)')
    parser.add_argument('--threshold-pixel', type=int, default=0.05,
                        help='The threshold of pixel differences (1st stage)')
    parser.add_argument('--boundaries', type=str, default='105,120,675,885',
                        help='The boundaries of slides: x_topleft, y_topleft, x_bottomright, y_bottomright')
    parser.add_argument('--edit-distance-threshold', type=float, default=0.6,
                        help='The maximum (Levenshtein_distance(s_{i}, s_{i+1}) / s_{i+1}) '
                             'between two slides s_{i} and s_{i+1} within a segment (2nd stage)')
    parser.add_argument('--min-interval', type=int, default=20,
                        help='The minimum time interval of a segment (in seconds)')
    parser.add_argument('--num-frame-forward', type=int, default=10,
                        help='The number of frames we look forward to determine the boundaries (2nd stage)')
    parser.add_argument('--bi-threshold', type=int, default=100,
                        help='The threshold for image binarization (0, 255), 0->black and 1->white (2nd stage)')

    opt = parser.parse_args()
    cutpoints = video_segmentation(opt)
    for idx, cutpoint in enumerate(cutpoints):
        print(idx + 1, '\t', cutpoint)
