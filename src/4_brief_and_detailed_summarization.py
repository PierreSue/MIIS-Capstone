import json
import os

from src.models.summarization.brief_summary.abstractive_sum import brief_summary
from src.models.summarization.detailed_summary.bart_inference import detailed_summary
from src.conf import DATA_DIR, VIDEO_ID


def get_summaries(input_path: str, output_path: str):
    # read data
    with open(input_path, 'r', encoding='utf8') as json_file:
        data = json.load(json_file)
    segments = data['segments']
    assert len(segments) > 0  # not support empty file

    # prepare src for summary model
    src = [info['transcript_corrected'] for info in segments]
    
    # get summary result
    brief_summaries = brief_summary(src)
    detailed_summaries = detailed_summary(src)
    assert len(brief_summaries) == len(detailed_summaries) == len(segments)

    # write result
    for i in range(len(segments)):
        segments[i]['summary_brief'] = brief_summaries[i]
        segments[i]['summary_detailed'] = detailed_summaries[i]

    with open(output_path, 'w', encoding='utf8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


def run():
    input_path = os.path.join(DATA_DIR, "results", VIDEO_ID + '.json')
    output_path = os.path.join(DATA_DIR, "results", VIDEO_ID + '.json')
    get_summaries(input_path, output_path)
    print(f'Results have been saved to {output_path}')


if __name__ == "__main__":
    run()



# def get_summaries(input_path: str, output_path: str,trained_path):
#     # read data
#     with open(input_path, 'r', encoding='utf8') as json_file:
#         segments = json.load(json_file)
#     # segments = data['segments']
#     assert len(segments) > 0  # not support empty file

#     # prepare src for summary model
#     src = [info['transcript-corrected'] for info in segments]
    
#     # get summary result
#     brief_summaries = brief_summary(src)
#     # write result
#     for i in range(len(segments)):
#         segments[i]['summary_brief'] = brief_summaries[i]
#         if "summary_detailed" in segments[i]:
#             segments[i].pop("summary_detailed")

#     with open(output_path, 'w', encoding='utf8') as json_file:
#         json.dump(segments, json_file, indent=4, ensure_ascii=False)

# if __name__ == "__main__":
#     input_path = "/home/zhihao/MIIS-Capstone/segmented_transcript-grammar_corrected.json"
#     output_path = "/home/zhihao/MIIS-Capstone_wzh/brief_model/peg_test_res.json"
#     # input_path = "/home/zhihao/MIIS-Capstone/11645_1.json"
#     # output_path = "/home/zhihao/MIIS-Capstone/peg_test_res_645.json"
#     # trained_path = None
#     trained_path = "/home/zhihao/MIIS-Capstone/results/train_model"
#     get_summaries(input_path,output_path,trained_path)