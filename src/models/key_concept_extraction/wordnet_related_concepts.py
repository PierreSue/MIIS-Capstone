# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: wordnet_related_concepts.py
@time: 11/28/21
"""
from typing import List, Dict

import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')


def get_related_concepts_wordnet(concept: str, lemma_of_hypo_hyper: bool = False) -> List[Dict[str, str]]:
    """

    Args:
        concept: query concept
        lemma_of_hypo_hyper: whether to use all lemmas of each hyponyms/hypernyms, False by default

    Returns:
        [
            {
                "concept_name": "hit",
                "relation": "synonyms"
            },
            {
                "concept_name": "contact",
                "relation": "hypernyms"
            },
            ...
        ]

    """
    concept = concept.lower()
    related_concepts = []

    syn_arr = wordnet.synsets(concept.replace(' ', '_'))
    for syno in syn_arr:
        if syno.name().split('.')[1] != 'n':
            continue
        for lemma in syno.lemma_names():
            lemma = lemma.replace('_', ' ')
            if lemma == concept:
                continue
            if concept.endswith('es') and lemma + 'es' == concept:
                continue
            if concept.endswith('s') and lemma + 's' == concept:
                continue
            related_concepts.append({
                'concept_name': lemma,
                'relation': 'synonyms'
            })
        for hypo in syno.hyponyms():
            if lemma_of_hypo_hyper:
                for lemma in hypo.lemma_names():
                    related_concepts.append({
                        'concept_name': lemma.replace('_', ' '),
                        'relation': 'hyponyms'
                    })
            else:
                related_concepts.append({
                    'concept_name': hypo.name().split('.')[0].replace('_', ' '),
                    'relation': 'hyponyms'
                })
        for hyper in syno.hypernyms():
            if lemma_of_hypo_hyper:
                for lemma in hyper.lemma_names():
                    related_concepts.append({
                        'concept_name': lemma.replace('_', ' '),
                        'relation': 'hypernyms'
                    })
            else:
                related_concepts.append({
                    'concept_name': hyper.name().split('.')[0].replace('_', ' '),
                    'relation': 'hypernyms'
                })

    return related_concepts


def main():
    related_concepts = get_related_concepts_wordnet('collision', lemma_of_hypo_hyper=True)
    print('=' * 10, 'collision (lemma_of_hypo_hyper = True)', '=' * 10)
    print('Concept\tRelation')
    for rel_concept in related_concepts:
        print(f"{rel_concept['concept_name']}\t{rel_concept['relation']}")

    related_concepts = get_related_concepts_wordnet('collision')
    print('=' * 10, 'collision (lemma_of_hypo_hyper = False)', '=' * 10)
    print('Concept\tRelation')
    for rel_concept in related_concepts:
        print(f"{rel_concept['concept_name']}\t{rel_concept['relation']}")

    related_concepts = get_related_concepts_wordnet('solar_calendar', lemma_of_hypo_hyper=True)
    print('=' * 10, 'solar_calendar (lemma_of_hypo_hyper = True)', '=' * 10)
    print('Concept\tRelation')
    for rel_concept in related_concepts:
        print(f"{rel_concept['concept_name']}\t{rel_concept['relation']}")

    related_concepts = get_related_concepts_wordnet('solar_calendar')
    print('=' * 10, 'solar_calendar (lemma_of_hypo_hyper = False)', '=' * 10)
    print('Concept\tRelation')
    for rel_concept in related_concepts:
        print(f"{rel_concept['concept_name']}\t{rel_concept['relation']}")


if __name__ == '__main__':
    main()
