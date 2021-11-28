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


def transcript_clean(transcript, max_length):
    doc = pos_tagger(transcript.replace('\n', ' '))

    queue, candidates = [], set()
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.pos not in {"PRON", "PROPN", 'DET', 'ADP', 'INTJ', 'ADV', 'AUX', 'CONJ', 'VERB', 'X'} and \
               word.text not in stopwords and not_has_numbers_punct(word.text) and len(word.text) > 2:
                    queue.append(word.text.lower())
            else:
                for i in range(len(queue)):
                    for length in range(1, max_length+1):
                        candidates.add(' '.join(queue[i:i+length]).lower())
                queue = []
    return candidates
    

def search_key_concepts(entry_id, candidates):
    output = []
    history = set()

    for candidate in candidates:
        try:
            # title, content, url
            wikipage = wikipedia.page(candidate)
            if not p.compare(candidate, wikipage.title.lower()):
                raise Exception()
            output.append({'Concept': wikipage.title.lower(), 'Summary': wikipage.summary, 'Content': wikipage.content, 'URL': wikipage.url})
        except:
            pass

    return entry_id, output

def multi_search_wrapper(args):
   return search_key_concepts(*args)