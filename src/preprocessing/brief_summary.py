import json

from src.models.summarization.brief_summary.abstractive_sum import brief_summary

def get_brief_summary(input_path,output_path):
    #read data
    with open(input_path,'r',encoding='utf8') as f:
        infos = json.load(f)    
    assert len(infos) > 0 #not support empty file

    #prepare src for summary model
    src = []
    for info in infos:
        src.append(info['transcript-corrected'])
    
    #get summary result
    res = brief_summary(src)
    assert len(res) == len(infos)

    #write result
    for i in range(len(infos)):
        infos[i]['brief-summary'] = res[i]

    with open(output_path,'w',encoding='utf8') as f2:
        json.dump(infos,f2,indent=4,ensure_ascii=False)


if __name__ =="__main__":
    input_path = "/home/zhihao/MIIS-Capstone_wzh/src/models/summarization/brief_summary/segmented_transcript-grammar_corrected.json"
    output_path = input_path.replace("segmented_transcript-grammar_corrected","segemented_brief-summary")
    get_brief_summary(input_path,output_path)