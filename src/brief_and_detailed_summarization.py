import json

from src.models.summarization.brief_summary.abstractive_sum import brief_summary
from src.models.summarization.detailed_summary.bart_inference import detailed_summary


def get_summaries(input_path: str, output_path: str):
    # read data
    with open(input_path, 'r', encoding='utf8') as f:
        segments = json.load(f)
    assert len(segments) > 0  # not support empty file

    # prepare src for summary model
    src = [info['transcript-corrected'] for info in segments]
    
    # get summary result
    brief_summaries = brief_summary(src)
    detailed_summaries = detailed_summary(src)
    assert len(brief_summaries) == len(detailed_summaries) == len(segments)

    # write result
    for i in range(len(segments)):
        segments[i]['summary-brief'] = brief_summaries[i]
        segments[i]['summary-detailed'] = detailed_summaries[i]

    with open(output_path, 'w', encoding='utf8') as f2:
        json.dump(segments, f2, indent=4, ensure_ascii=False)


def run():
    input_path = '/home/jingrong/capstone/data/result-grammar_corrected.json'
    output_path = '/home/jingrong/capstone/data/result-summary.json'
    get_summaries(input_path, output_path)


if __name__ == "__main__":
    run()
