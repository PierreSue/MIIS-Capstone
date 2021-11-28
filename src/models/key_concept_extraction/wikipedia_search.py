import json
import string
import wikipedia
from multiprocessing import Pool

import warnings
warnings.catch_warnings()
warnings.simplefilter("ignore")

import inflect
p = inflect.engine()

import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

import stanza
stanza.download('en')
pos_tagger = stanza.Pipeline('en', processors='tokenize,pos')

def not_has_numbers_punct(inputString):
    return not any(char.isdigit() or char in string.punctuation+'"'+'—' for char in inputString)


def transcript_clean(transcript):
    doc = pos_tagger(transcript.replace('\n', ' '))

    revised_transcript = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.pos not in {"PRON", "PROPN", 'DET', 'ADP', 'INTJ', 'ADV', 'AUX', 'CONJ', 'VERB', 'X'} and word.text not in stopwords:
                if not_has_numbers_punct(word.text) and len(word.text) > 2:
                    revised_transcript.append(word.text.lower())
    return revised_transcript
    

def search_key_concepts(entry_id, transcript, max_length):
    output = []
    history = set()

    i = 0
    for i in range(len(transcript)):
        for length in range(1, max_length+1):
            phrase = ' '.join(transcript[i:i+length]).lower()
            if phrase not in history:
                history.add(phrase)
                try:
                    # title, content, url
                    wikipage = wikipedia.page(phrase)
                    if not p.compare(phrase, wikipage.title.lower()):
                        raise Exception()
                    output.append({'Concept': wikipage.title.lower(), 'Summary': wikipage.summary, 'Content': wikipage.content, 'URL': wikipage.url})
                except:
                    pass
    return entry_id, output

def multi_search_wrapper(args):
   return search_key_concepts(*args)