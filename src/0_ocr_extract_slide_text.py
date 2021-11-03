# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: 0_ocr_extract_slide_text.py
@time: 11/1/21
"""
import json
import os
from typing import List, Dict

import numpy as np

from src.utils import BI_THRESHOLD_OCR, add_key_value_into_jsonfile
from src.conf import DATA_DIR, VIDEO_ID

try:
    from PIL import Image
except ImportError:
    import Image

from src.models.segmentation.utils import OCRTextExtractor


def extract_text(slide_image_dir: str, bi_threshold: int) -> List[Dict[str, str]]:
    slides_imgs = []
    file_list = sorted(os.listdir(slide_image_dir), key=lambda x: int(x.split('.')[-2].split('_')[-1]))
    for img_file in file_list:
        if not img_file.endswith('.jpg'):
            print(f'Skip file {img_file}, only JPG files are supported')
            continue
        img = Image.open(os.path.join(slide_image_dir, img_file))
        slides_imgs.append(np.array(img))

    ocr_extractor = OCRTextExtractor()
    ocr_text_list = ocr_extractor.batch_ocr(np.array(slides_imgs), bi_threshold=bi_threshold)
    return ocr_text_list


def run():
    slide_image_dir = os.path.join(DATA_DIR, "slides", VIDEO_ID)
    output_file = os.path.join(DATA_DIR, "results", VIDEO_ID + '.json')
    ocr_text_list = extract_text(slide_image_dir, BI_THRESHOLD_OCR)

    add_key_value_into_jsonfile(output_file, key='slides_ocr_texts', value=ocr_text_list)
    print(f'Results have been saved to {output_file}')


if __name__ == '__main__':
    run()
