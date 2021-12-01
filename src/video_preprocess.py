# ==============================================================
# with open(opt.output_path, 'r', encoding='utf8') as json_file:
#     data = json.load(json_file)
# ==============================================================
import os
import sys
import json
import gdown
import shutil
import string
import argparse
import importlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from p_tqdm import p_map

from punctuator import Punctuator
from utils import segment_sentence
from gingerit.gingerit import GingerIt
from models.punctuation_restoration import punctuation_restoration_v2
from models.segmentation import video_segmentation, OCRTextExtractor
from models.transcript_preprocess import transcript_preprocess
from models.summarization import brief_summary, detailed_summary
from models.key_concept_extraction import multi_search_wrapper, DocSim, transcript_clean, search_key_concepts, get_related_concepts_wordnet


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Video Segmentation (1st stage)
    parser.add_argument('--video-path', type=str, help='The input video path')
    parser.add_argument('--output-path', type=str, help='The output json file path')
    parser.add_argument('--image-dir', type=str, help='The output image directory')
    parser.add_argument('--interval', type=int, default=2,
                        help='The interval between each image in seconds. (An image/N sec)')
    parser.add_argument('--threshold-pixel', type=int, default=0.05,
                        help='The threshold of pixel differences (1st stage)')
    parser.add_argument('--boundaries', type=str, default='105,120,675,885',
                        help='The boundaries of slides: x_topleft, y_topleft, x_bottomright, y_bottomright')
    
    # Video Segmentation (2nd stage) & OCR Extraction
    parser.add_argument('--edit-distance-threshold', type=float, default=0.6,
                        help='The maximum (Levenshtein_distance(s_{i}, s_{i+1}) / s_{i+1}) '
                             'between two slides s_{i} and s_{i+1} within a segment (2nd stage)')
    parser.add_argument('--min-interval', type=int, default=60,
                        help='The minimum time interval of a segment (in seconds)')
    parser.add_argument('--num-frame-forward', type=int, default=10,
                        help='The number of frames we look forward to determine the boundaries (2nd stage)')
    parser.add_argument('--bi-threshold', type=int, default=100,
                        help='The threshold for image binarization (0, 255), 0->black and 1->white (2nd stage)')

    # Transcript Preprocess
    parser.add_argument('--caption-path', type=str, help='The input caption path')
    parser.add_argument('--weight-path', type=str, help='The input file path (Roberta-large)', default='./roberta-large-en.pt')

    # keyword Extraction
    parser.add_argument('--max-words', type=int, help='The max number of words in key concepts', default=3)
    parser.add_argument('--doc-weight-path', type=str, help='The input file path (Doc Similarity)', default='./GoogleNews-vectors-negative300.bin')
    parser.add_argument('--concept-threshold', type=float, help='The threshold of Doc Similarity', default=0.8)
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_arguments()

    # Initialization
    if not os.path.exists(os.path.dirname(opt.output_path)):
        os.makedirs(os.path.dirname(opt.output_path))
    if os.path.exists(opt.image_dir):
        # val = input("Remove all the contents in the directory: {} | [Y]/N : ".format(opt.image_dir))
        # if val in {'Y', 'y', ''}:
        #     shutil.rmtree(opt.image_dir)
        #     os.makedirs(opt.image_dir)
        shutil.rmtree(opt.image_dir)
        os.makedirs(opt.image_dir)
    else:
        os.makedirs(opt.image_dir)


    # Spec
    ID = os.path.splitext(os.path.basename(opt.output_path))[0]
    Lecturer = ID.split('_')[0]
    data = { 'ID': ID, 'name': '', 'description': '', \
             'lecturerName': '', 'lecturerAvatar': '{}.png'.format(Lecturer), \
             'time': '', 'youtube_link': '', 'segments': []}


    # Video Segmentation
    slides_imgs = []
    valid_boundaries, segments, images, time = video_segmentation(opt)
    data['time'] = time
    for i, (segmentA, segmentB) in enumerate(zip(segments[:-1], segments[1:])):
        slides_imgs.append(images[(segmentA+segmentB)//2])
        im = Image.fromarray(images[(segmentA+segmentB)//2])
        im.save(os.path.join(opt.image_dir, '{}.jpg'.format(i)))

    ## Last Image
    slides_imgs.append(images[(segments[-1]+len(images))//2])
    im = Image.fromarray(images[(segments[-1]+len(images))//2])
    im.save(os.path.join(opt.image_dir, '{}.jpg'.format(len(segments)-1)))
    print('Video Segmentation Finished.')


    # OCRTextExtractor (title, text)
    ocr_extractor = OCRTextExtractor()
    ocr_text_list = ocr_extractor.batch_ocr(np.array(slides_imgs), bi_threshold=opt.bi_threshold)
    assert len(ocr_text_list) == len(valid_boundaries)+1
    del ocr_extractor, images, slides_imgs
    print('OCR Finished.')

    ## Generate and Save Temporal Outputs
    for valid_boundary, ocr_text in zip(['00:00:00']+valid_boundaries, ocr_text_list):
        segment = {'start_timestamp': valid_boundary, 'title': ocr_text['title'], 'text': ocr_text['text']}
        data['segments'].append(segment)
    with open(opt.output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    # Transcript Preprocess
    transcripts = transcript_preprocess(valid_boundaries, opt.caption_path)
    transcripts = punctuation_restoration_v2(transcripts, opt.weight_path)
    print('Transcripts Preprocess Finished.')
    
    
    # Grammar Correction
    parser = GingerIt()
    corrected_transcripts = []
    for i, (segment, transcript) in tqdm(enumerate(zip(data['segments'], transcripts)), total=len(data['segments']), desc='Grammar Correction', ncols=80):
        corrected_transcript = []
        for sentence in segment_sentence(transcript):
            try: # The sentence cannot be longer than 300 chars.
                corrected_transcript.append(parser.parse(sentence)['result'])
            except:
                corrected_transcript.append(sentence)
        corrected_transcript = ' '.join(corrected_transcript)
        segment['transcript'] = transcript
        segment['transcript-corrected'] = corrected_transcript
        corrected_transcripts.append(corrected_transcript)
        data['segments'][i] = segment
    with open(opt.output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    del parser
    print('Grammar Correction Finished.')


    # Brief and detailed summarization
    # Cuda errors might happen if there are too many segmentations (Esp. Detailed)
    batch_size = 15
    brief_summaries, detailed_summaries = [], []
    for i in range(0, len(corrected_transcripts), batch_size):
        brief_summaries += brief_summary(corrected_transcripts[i:i+batch_size])
        detailed_summaries += detailed_summary(corrected_transcripts[i:i+batch_size])

    for i, (segment, brief, detailed) in enumerate(zip(data['segments'], brief_summaries, detailed_summaries)):
        segment['summary_brief'] = brief
        segment['summary_detailed'] = detailed
        data['segments'][i] = segment
    with open(opt.output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    del brief_summaries, detailed_summaries, corrected_transcripts
    print('Brief/Detailed Summarization Finished.')


    # Keyword Extraction
    for i in range(len(data['segments'])):
        data['segments'][i]['key_concepts'] = {}

    texts = []
    for i, entry in enumerate(data['segments']):
        # texts += [(i, transcript_clean(entry['text']), opt.max_words)]
        # texts += [(i, transcript_clean(entry['transcript-corrected']), opt.max_words)]
        texts += [(i, transcript_clean(entry['text'], opt.max_words))]
        texts += [(i, transcript_clean(entry['transcript-corrected'], opt.max_words))]

    num_cores = 4
    key_concepts = p_map(multi_search_wrapper, texts, num_cpus=num_cores)

    ds = DocSim(opt.doc_weight_path)
    for entry_id, wikicontent in tqdm(key_concepts, total=len(key_concepts), desc='Keyword Extraction', ncols=80):
        source_doc = data['segments'][entry_id]['transcript-corrected']
        target_docs = [content['Content'] for content in wikicontent]
        sim_scores = ds.calculate_similarity(source_doc, target_docs)
        for sim_score, content in zip(sim_scores, wikicontent):
            if sim_score['score'] > opt.concept_threshold:
                data['segments'][entry_id]['key_concepts'][content['Concept']] = {'Score': str(sim_score['score']), 'Summary': content['Summary'], 'URL': content['URL'], 'Related_Concepts':get_related_concepts_wordnet(content['Concept'])}
    
    with open(opt.output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print('Keyword Extraction Finished.')