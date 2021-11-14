# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: utils.py
@time: 11/13/21
"""
from typing import List, Dict, Set
from xml.etree.ElementTree import fromstring

import requests
import stanza
from urllib3.exceptions import InsecureRequestWarning
from xmljson import parker

# Suppress only the single warning from urllib3 needed.
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

stanza.download('en')
NLP_PROCESSOR = stanza.Pipeline('en', processors='tokenize,pos')


def get_nouns(text: str, lower: bool) -> Set[str]:
    if lower:
        text = text.lower()
    doc = NLP_PROCESSOR(text)

    nouns = set()
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.pos == "NOUN":
                nouns.add(word.text)
    return nouns


def query_in_dbpedia(query: str, incl_desc: bool = False) -> List[Dict]:
    url = f'https://lookup.dbpedia.org/api/search?query={query}'
    response = requests.get(url, verify=False)
    response_json = parker.data(fromstring(response.text))
    if response_json is None:
        return []
    filtered_response_json = []
    if not isinstance(response_json['Result'], list):
        results = [response_json['Result']]
    else:
        results = response_json['Result']
    for item in results[:3]:
        if not item['Label']:
            continue
        if incl_desc and item['Description']:
            item_content = item['Label'].lower() + '\n' + item['Description'].lower()
        else:
            item_content = item['Label'].lower()
        if (query in item_content or
                (query[-1] == 's' and query[:-1] in item_content) or
                (query[-2:] == 'es' and query[:-2] in item_content)):
            filtered_response_json.append(item)
    return filtered_response_json


if __name__ == '__main__':
    query_in_dbpedia("phonemes", True)
