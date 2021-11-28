import os
import gdown
from tqdm import tqdm
from punctuator import Punctuator

def punctuation_restoration_v2(transcripts, weight_file):
    # Download the Roberta-large model file if the model file does not exist.
    if not os.path.exists(weight_file):
        url = 'https://drive.google.com/uc?id=1btpRPOloqTToW2XmfSEGtBh9Gzj9g-0G'
        gdown.download(url, weight_file)

    # Model and Tokenizer
    p = Punctuator(weight_file)

    restored_transcript_segments = []
    for transcript in tqdm(transcripts, total=len(transcripts), desc='transcript', ncols=80):
        try:
            capitalized_segment = p.punctuate(transcript)
        except:
            capitalized_segment = transcript
            print(capitalized_segment)
            print("[WARNING] This transcript is not punctuatable.")
        restored_transcript_segments.append(capitalized_segment)

    del p
    return restored_transcript_segments

if __name__ == '__main__':
    transcripts = ['hello it is a test', 'this is complex i promise please make it easier to use']
    restored_transcript_segments = punctuation_restoration_v2(transcripts, './model.pcl')
    print(restored_transcript_segments)