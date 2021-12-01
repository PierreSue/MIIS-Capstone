import os
import sys
import json
import heapq
from collections import defaultdict

from flask_cors import CORS
from flask import (Flask, render_template, flash,
                    request, jsonify, Markup)
from search_engine import Segs2BM25, SearchByQuery


application = Flask(__name__)
CORS(application)

@application.before_first_request
def startup():
    global OutputPath, JsonList, BM25Model, SegmentMap

    try:
        OutputPath = sys.argv[1]
    except:
        OutputPath = '../output'
    print("This is the path to the data directory: {} /jsons /imgs".format(OutputPath))

    jsonpath = os.path.join(OutputPath, 'jsons')
    jsonfiles = [os.path.join(jsonpath, f) for f in os.listdir(jsonpath) if f.endswith('.json')]
    
    JsonList = {}
    for jsonfile in jsonfiles:
        with open(jsonfile, 'r', encoding='utf8') as json_file:
            ID = os.path.splitext(os.path.basename(jsonfile))[0]
            JsonList[ID] = json.load(json_file)
    
    SegmentMap, BM25Model = {}, {}
    for ID, content in JsonList.items():
        SegmentMap[ID] = {}
        for segment in content['segments']:
            references = [segment["transcript-corrected"]]
            for concept, info in segment['key_concepts'].items():
                references.append(info['Summary'])
            SegmentMap[ID][segment['start_timestamp']] = references
        BM25Model[ID] = Segs2BM25(SegmentMap[ID])


# Input  = {'topn': top N keywords (Default 3)}
# Output = [lectures{'ID', 'name', 'description', 'lecturerName', 'lecturerAvatar', 'time', 'keywords'}]
@application.route('/retrieve_gallery', methods=['POST', 'GET'])
def retrieve_gallery():
    topn = int(request.args.get('topn', default=3))

    lectures = []
    for ID, content in JsonList.items():
        conceptMap = defaultdict(float)
        for segment in content['segments']:
            for concept, info in segment['key_concepts'].items():
                bias = (len(concept.split())-1)*0.1
                conceptMap[concept] = max(conceptMap[concept], float(info['Score'])+bias)
        concepts = heapq.nlargest(topn, conceptMap.keys(), key=lambda k: conceptMap[k])

        lectures.append({'ID': content['ID'], 'name': content['name'], 'description': content['description'], \
                         'lecturerName': content['lecturerName'], \
                         'lecturerAvatar': os.path.join(OutputPath, 'imgs', content['lecturerAvatar']), \
                         'time': content['time'], 'keywords': concepts, 'videoURL': concepts['youtube_link']})
    
    return jsonify(lectures)


# Input  = {'ID': lecture_id, 'Query': input_query, 'topn': N (Default 3)}
# Output = {'scores': scores, 'topn': top N segments based on the query}
@application.route('/search_engine', methods=['POST', 'GET'])
def search_engine():
    ID = request.args.get('ID')
    Query = request.args.get('Query')
    topn = int(request.args.get('topn', default=min(3, len(SegmentMap[ID]))))

    if len(Query) == 0 or Query:
        return

    results = SearchByQuery(Query, BM25Model[ID], topn=topn)
    return jsonify({'scores': results[0], 'topn': results[1]})


# Input  = {'ID': lecture_id}
# Output = [segments{'start_timestamp', 'transcript_corrected'}]
@application.route('/transcript', methods=['POST', 'GET'])
def transcript():
    ID = request.args.get('ID')

    segments = []
    for segment in JsonList[ID]['segments']:
        segments.append({'transcript_corrected': segment['transcript-corrected']})
    return jsonify(segments)


# Input  = {'ID': lecture_id}
# Output = [segments{'summary_brief', 'summary_detailed'}]
@application.route('/summary', methods=['POST', 'GET'])
def summary():
    ID = request.args.get('ID')

    segments = []
    for segment in JsonList[ID]['segments']:
        segments.append({'start_timestamp': segment['start_timestamp'], 'summary_brief': segment['summary_brief'], 'summary_detailed': segment['summary_detailed']})
    return jsonify(segments)


# Input  = {'ID': lecture_id, topn: top N keywords for each segment(Default 2)}
# Output = [segments{'keywords'}]
@application.route('/keywords', methods=['POST', 'GET'])
def keywords():
    ID = request.args.get('ID')
    topn = int(request.args.get('topn', default=2))

    segments = []
    for segment in JsonList[ID]['segments']:
        conceptMap = defaultdict(float)
        for concept, info in segment['key_concepts'].items():
            bias = (len(concept.split())-1)*0.1
            conceptMap[concept] = max(conceptMap[concept], float(info['Score'])+bias)

        keywords = {}
        concepts = heapq.nlargest(topn, conceptMap.keys(), key=lambda k: conceptMap[k])
        for concept in concepts:
            keywords[concept] = {"Summary": segment['key_concepts'][concept]["Summary"], "URL": segment['key_concepts'][concept]["URL"], "Related_Concepts": segment['key_concepts'][concept]["Related_Concepts"]}
        segments.append({'keywords': keywords})
    return jsonify(segments)

if __name__ == '__main__':
    application.debug = False
    application.run(host='0.0.0.0')