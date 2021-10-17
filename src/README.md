# Procedures
## Using YouTube Subtitles
0. Get segmentation boundaries: `models/segmentation/video_segmentation.py`
1. Merge and segment subtitles, plus punctuation and capitalization restoration: `transcript_preproc_on_youtube_asr.py`
2. Correct grammatical errors: `grammar_correction.py`
3. Get brief summary (using PEGASUS) and detailed summary (using BART) for each segment: `brief_and_detailed_summarization.py`

## Using Speech-to-Text API on Google Cloud
0. Get segmentation boundaries: `models/segmentation/video_segmentation.py`
1. Follow step 2 to step 5 on this [Tutorial: Using the Speech-to-Text API with Python](https://codelabs.developers.google.com/codelabs/cloud-speech-text-python3)
to setup environment and authenticate API requests
2. ASR using Speech-to-Text API on Google Cloud: `gc_stt_asr.py`
3. Merge and segment subtitles, plus punctuation and capitalization restoration: `transcript_preproc_on_gc_stt.py`
4. Correct grammatical errors: `grammar_correction.py`
5. Get brief summary (using PEGASUS) and detailed summary (using BART) for each segment: `brief_and_detailed_summarization.py`

# Output File Format
```python
[
    {
        "start_timestamp": "00:00:00",
        "transcript": "Okay, xxxx vocal.",
        "transcript-corrected": "Okay, xxxx vocal.",
        "summary-brief": "blablabla",
        "summary-detailed": "blablabla. blablabla. blablabla."
    }  # one dict for each segment
]
```
