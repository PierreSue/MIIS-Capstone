import string
from rank_bm25 import BM25Okapi
# http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf

remove_punc = str.maketrans("", "", string.punctuation)

def Segs2BM25(SegmentMap):
    corpus, documents, keys = [], [], []
    for key, docs in SegmentMap.items():
        documents.append(docs)

        document = ''
        for doc in docs:
            document += doc.translate(remove_punc)
        corpus.append(document)
        keys.append(key)
    
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return (bm25, documents, keys)


def SearchByQuery(Query, BM25, topn=1):
    bm25, documents, keys = BM25

    Query = Query.translate(remove_punc)
    tokenized_query = Query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)

    output, max_val = {}, 0.
    for score, docs, key in zip(doc_scores, documents, keys):
        output[key] = {'Score': score*100, "Documents": docs}
        max_val = max(score, max_val)
    
    if max_val != 0.:
        for key, value in output.items():
            value['Score'] /= max_val
            output[key] = value
    
    return output, bm25.get_top_n(tokenized_query, keys, n=topn)


if __name__ == "__main__":
    # Topn: The top N Segments for Retrieval
    # Query: String
    # SegmentMap: Key -> List of Strings
    Topn = 1
    Query = "Speech Recognition"
    SegmentMap = {
        'abc': ["Hello there good man!"],
        'def': ["It is quite windy in London", "How is the weather today?"],
        'ghi': ["How are you?", "This is Speech Processing.", "Hello."],
    }

    # BM25: (THe BM25 Model, Documents, Keys)
    # results: (Key -> {"Score", "Documents"}, Top N Keys)
    BM25 = Segs2BM25(SegmentMap)
    results = SearchByQuery(Query, BM25, topn=Topn)

    print(results)

