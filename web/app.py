from flask_cors import CORS
from flask import (Flask, render_template, flash,
                    request, jsonify, Markup)
from web.search_engine import Segs2BM25, SearchByQuery

import json

application = Flask(__name__)
CORS(application)

@application.before_first_request
def startup():

    global BM25Model, SegmentMap
    
    with open('/home/pierre/Capstone/Data/segmented_transcript-grammar_corrected.json', 'r') as fp:
        JsonList = json.load(fp)
    
    SegmentMap = {}
    for segment in JsonList:
        SegmentMap[segment['start_timestamp']] = [segment["transcript-corrected"]]
    BM25Model = Segs2BM25(SegmentMap)


@application.route('/search_engine', methods=['POST', 'GET'])
def search_engine():
    Query = request.args.get('Query')
    topn = request.args.get('topn', default=min(3, len(SegmentMap)))

    print(Query)

    results = SearchByQuery(Query, BM25Model, topn=topn)
    return jsonify({'scores': results[0], 'topn': results[1]})


@application.route('/transcript', methods=['POST', 'GET'])
def transcript():
    video_idx = request.form.get('video_idx')
    assert video_idx is not None
    #the path should be constructed according to the video_idx
    smry_path = f"/mnt/capstone/data/transcript-summary_{video_idx}.json"
    res = []
    with open(smry_path,'r',encoding='utf8') as f:
        infos = json.load(f)   
    assert(len(infos) >= 1)
    for i,info in enumerate(infos):
        info.pop("transcript")
        info.pop("summary-brief")
        info.pop("summary-detailed")
        res.append(info)
    print(jsonify(res))
    return jsonify(res)

@application.route('/summary', methods=['POST', 'GET'])
def summary():
    video_idx = request.form.get('video_idx')
    assert video_idx is not None
    #the path should be constructed according to the video_idx
    smry_path = f"/mnt/capstone/data/transcript-summary_{video_idx}.json"
    res = []
    with open(smry_path,'r',encoding='utf8') as f:
        infos = json.load(f)   
    assert(len(infos) >= 1)
    for i,info in enumerate(infos):
        info.pop("transcript")
        info.pop("transcript-corrected")
        res.append(info)
    return jsonify(res)