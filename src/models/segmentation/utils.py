# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: utils.py
@time: 9/30/21
"""
from typing import List, Dict, Tuple

import cv2
import numpy as np
import pytesseract
from Levenshtein import distance as edit_distance

try:
    from PIL import Image
except ImportError:
    import Image


class OCRTextExtractor(object):
    """
    One OCRTextExtractor object for one video only
    """

    def __init__(self):
        self.second_to_text_info = dict()  # Avoid duplicate calculations

    @staticmethod
    def rgb_to_binary(image: np.ndarray, bi_threshold: int) -> Image:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(image_gray, bi_threshold, 255, cv2.THRESH_BINARY)
        return Image.fromarray(binary_image)

    def batch_ocr(self, batch_image: np.ndarray, batch_second: List[int], bi_threshold: int) -> List[Dict[str, str]]:
        assert len(batch_image) == len(batch_second)
        extracted_text_lst = []
        for image, second in zip(batch_image, batch_second):
            if second in self.second_to_text_info:
                text_info = self.second_to_text_info[second]
            else:
                binary_image = self.rgb_to_binary(image, bi_threshold)
                text = pytesseract.image_to_string(binary_image, lang='eng')
                text_info = {
                    'title': text.strip().split('\n\n')[0].strip(),
                    'text': text.strip(),
                    'original_text': text
                }
                self.second_to_text_info[second] = text_info
            extracted_text_lst.append(text_info)
        return extracted_text_lst


def calculate_diff_ratio(prev_text: str, cur_text: str) -> float:
    return edit_distance(prev_text, cur_text) / (len(cur_text) + 1e-6)


def is_segment_boundary(batch_second_text: List[Tuple[int, Dict[str, str]]],
                        cur_idx: int,
                        last_boundary: int,
                        threshold_ratio: float,
                        min_interval: int) -> bool:
    assert 0 < cur_idx < len(batch_second_text) - 1, f'At least 3 frames forward, 1 frame backward | cur_idx={cur_idx}'
    assert len(batch_second_text) >= 5, f'At least 3 frames forward, 1 frame backward | width={len(batch_second_text)}'

    cur_second = batch_second_text[cur_idx][0]
    cur_text = batch_second_text[cur_idx][1]['text']
    last_text = batch_second_text[cur_idx - 1][1]['text']
    next_text = batch_second_text[cur_idx + 1][1]['text']

    diff_ratio_prev_cur = calculate_diff_ratio(last_text, cur_text)
    diff_ratio_prev_next = calculate_diff_ratio(last_text, next_text)
    diff_ratios_prevs = [calculate_diff_ratio(batch_second_text[i][1]['text'], cur_text) for i in range(cur_idx - 1)]

    # Regarded as a boundary iff:
    # 1. diff_ratio = Levenshtein_distance(s_{i}, s_{i+1}) / s_{i+1} > threshold_ratio (incremental slides)
    # 2. segment_interval > min_interval (skip too short segments)
    # 3. diff_ratio between the previous frame and the next frame  > threshold_ratio
    # 4. min(diff_ratios of previous K frames) > threshold_ratio
    if diff_ratio_prev_cur > threshold_ratio and cur_second - last_boundary > min_interval:
        if diff_ratio_prev_next <= threshold_ratio:
            return False
        # use second smallest instead of minimum to take into account jumping page forward and backward
        if np.partition(diff_ratios_prevs, 1)[1] <= threshold_ratio:
            return False
        return True
    return False
