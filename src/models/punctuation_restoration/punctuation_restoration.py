import os
import re
import torch
import gdown

from tqdm import tqdm
from transformers import *
from .src import DeepPunctuation, DeepPunctuationCRF

import nltk
nltk.download('punkt')

def segment_sentence(paragraph):
    sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sen_tokenizer.tokenize(paragraph)
    return sentences


def punctuation_restoration(transcripts, weight_file, language='en', use_gpu=True):
    # Download the Roberta-large model file if the model file does not exist.
    if not os.path.exists(weight_file):
        url = 'https://drive.google.com/uc?id=17BPcnHVhpQlsOTC8LEayIFFJ7WkL00cr'
        gdown.download(url, weight_file)

    # Model and Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    token_idx = {'START_SEQ': 0, 'PAD': 1, 'END_SEQ': 2, 'UNK': 3}
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    deep_punctuation = DeepPunctuation('roberta-large', freeze_bert=False, lstm_dim=-1)
    deep_punctuation.to(device)

    # Load model parameters
    deep_punctuation.load_state_dict(torch.load(weight_file))
    deep_punctuation.eval()

    restored_transcript_segments = []
    for transcript in tqdm(transcripts, total=len(transcripts), desc='transcript', ncols=80):
        text = re.sub(r"[,:\-–.!;?]", '', transcript)
        words_original_case, words = text.split(),text.lower().split()

        result = ""
        word_pos, decode_idx, sequence_len = 0, 0, 256
        punctuation_map = {0: '', 1: ',', 2: '.' if language=='en' else '|', 3: '?'}   
        while word_pos < len(words):
            x, y_mask = [token_idx['START_SEQ']], [0]

            while len(x) < sequence_len and word_pos < len(words):
                tokens = tokenizer.tokenize(words[word_pos])
                if len(tokens) + len(x) >= sequence_len:
                    break
                else:
                    for i in range(len(tokens) - 1):
                        x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                        y_mask.append(0)
                    x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                    y_mask.append(1)
                    word_pos += 1
            x.append(token_idx['END_SEQ'])
            y_mask.append(0)
            if len(x) < sequence_len:
                x = x + [token_idx['PAD'] for _ in range(sequence_len - len(x))]
                y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
            attn_mask = [1 if token != token_idx['PAD'] else 0 for token in x]

            x = torch.tensor(x).reshape(1,-1)
            y_mask = torch.tensor(y_mask)
            attn_mask = torch.tensor(attn_mask).reshape(1,-1)
            x, attn_mask, y_mask = x.to(device), attn_mask.to(device), y_mask.to(device)

            with torch.no_grad():
                y_predict = deep_punctuation(x, attn_mask)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
            for i in range(y_mask.shape[0]):
                if y_mask[i] == 1:
                    result += words_original_case[decode_idx] + punctuation_map[y_predict[i].item()] + ' '
                    decode_idx += 1
        
        sentences = segment_sentence(result)
        capitalized_segment = ' '.join([sen.capitalize() for sen in sentences])
        restored_transcript_segments.append(capitalized_segment)

    return restored_transcript_segments

if __name__ == '__main__':
    transcripts = ['hello it is a test', 'this is complex i promise please make it easier to use']
    restored_transcript_segments = punctuation_restoration(transcripts, './roberta-large-en.pt')
    print(restored_transcript_segments)