# Quickstart

Note: `<VIDEO_ID>` below is a string of the unique ID of the lecture you'd like to process, e.g. "11692_1".

### 1. Data Collection
* Upload the .mp4 file named `<VIDEO_ID>.mp4` under the directory `/mnt/capstone/data/video`
* Upload the .sbv file (YouTube subtitle file) named `<VIDEO_ID>.sbv` under the directory `/mnt/capstone/data/youtube_asr`
* Make a folder `mkdir /mnt/capstone/data/slides/<VIDEO_ID>`, and upload image (only support .jpg) of each slide (named `xxx_<index>.jpg`) under this folder. Please note that these image files will be sorted according to the `<index>`.

### 2. Configuration
Update the variable `VIDEO_ID` (line #10) in `src/conf.py` with your `<VIDEO_ID>`. You are all set now.

### 3. Run with conda environments on the server (3.144.75.124):
```shell script
cd /path/to/MIIS-Capstone
export export PYTHONPATH="$PWD"

conda activate jingrong
python src/0_ocr_extract_slide_text.py
python src/1_segmentation.py
python src/2_transcript_preproc_on_youtube_asr.py
python src/3_grammar_correction.py
conda deactivate

conda activate summertime
python src/4_brief_and_detailed_summarization.py
conda deactivate

conda activate jingrong
python src/5_key_concept_extraction.py
```

The outputs will be saved to `/mnt/capstone/data/results/<VIDEO_ID>.json`.


# Output File Format

```python
{
    "slides_ocr_texts": [
        {
            "title": "Human Speech",
            "text": "Human Speech\n\n \n\n...",
            "original_text": "Human Speech\n\n \n\n...\n\f"
        },  # one dict for each slide image
        ...
    ],
    "segmentation_boundaries": [
        "00:01:16",
        "00:03:42",  # one string for each segment boundary
        ...
    ],
    "segments": [
        {
            "start_timestamp": "00:00:00",
            "transcript": "Okay, we might...",
            "transcript_corrected": "Okay, we might have...",
            "summary_brief": "Hello everyone...",
            "summary_detailed": "Today's lecture is...",
            "topk_concepts": [
                {
                    "concept": "speech",
                    "term_count_in_transcript_and_summary": 5
                },  # one dict for each concept for this segments
                ...
            ]
        },  # one dict for each segment
        ...
    ],
    "key_concepts": {
        "speech": {
            "term_count_in_slides": 8,
            "in_slide_title": true,
            "slide_indices": [
                0,
                8,
                21,
                24,
                25,
                26
            ],
            "dbpedia_results": [
                {
                    "Label": "Freedom of speech",
                    "URI": "http://dbpedia.org/resource/Freedom_of_speech",
                    "Description": "Freedom of speech is a principle that supports the freedom of an individual or a community to articulate their opinions and ideas without fear of retaliation, censorship, or legal sanction. The term \"freedom of expression\" is sometimes used synonymously but includes any act of seeking, receiving, and imparting information or ideas, regardless of the medium used.",
                    "Classes": {...},
                    "Categories": {...},
                    "Refcount": 65
                },  # one dict for each DBpedia lookup result
                ...
            ]
        },  # one dict for each concept
        ...
    },
    "topk_concepts": [
        {
            "concept": "speech",
            "score": 12
        },  # one dict for each top-k concepts for the lecture
        ...
    ]
}
```

# Procedures
## Using YouTube Subtitles
0. Extract texts in slides: `0_ocr_extract_slide_text.py`
1. Get segmentation boundaries: `1_segmentation.py`
2. Merge and segment subtitles, plus punctuation and capitalization restoration: `2_transcript_preproc_on_youtube_asr.py`
3. Correct grammatical errors: `3_grammar_correction.py`
4. Get brief summary (using PEGASUS) and detailed summary (using BART) for each segment: `4_brief_and_detailed_summarization.py`
5. Extract key concepts for the lecture and each segments: `5_key_concept_extraction.py`

## Using Speech-to-Text API on Google Cloud
0. Extract texts in slides: `0_ocr_extract_slide_text.py`
1. Get segmentation boundaries: `1_segmentation.py`
2. Follow step 2 to step 5 on this [Tutorial: Using the Speech-to-Text API with Python](https://codelabs.developers.google.com/codelabs/cloud-speech-text-python3)
to setup environment and authenticate API requests
3. ASR using Speech-to-Text API on Google Cloud: `gc_stt_asr.py`
4. Merge and segment subtitles, plus punctuation and capitalization restoration: `transcript_preproc_on_gc_stt.py`
5. Correct grammatical errors: `3_grammar_correction.py`
6. Get brief summary (using PEGASUS) and detailed summary (using BART) for each segment: `4_brief_and_detailed_summarization.py`
7. Extract key concepts for the lecture and each segments: `5_key_concept_extraction.py`
