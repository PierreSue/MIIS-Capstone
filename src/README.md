# Quick Start

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
        "00:03:42",
        ...
    ],
    "segments": [
        {
            "start_timestamp": "00:00:00",
            "transcript": "Okay, we might...",
            "transcript_corrected": "Okay, we might have...",
            "summary_brief": "Hello everyone...",
            "summary_detailed": "Today's lecture is..."
        },  # one dict for each segment
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

## Using Speech-to-Text API on Google Cloud
0. Extract texts in slides: `0_ocr_extract_slide_text.py`
1. Get segmentation boundaries: `1_segmentation.py`
2. Follow step 2 to step 5 on this [Tutorial: Using the Speech-to-Text API with Python](https://codelabs.developers.google.com/codelabs/cloud-speech-text-python3)
to setup environment and authenticate API requests
3. ASR using Speech-to-Text API on Google Cloud: `gc_stt_asr.py`
4. Merge and segment subtitles, plus punctuation and capitalization restoration: `transcript_preproc_on_gc_stt.py`
5. Correct grammatical errors: `3_grammar_correction.py`
6. Get brief summary (using PEGASUS) and detailed summary (using BART) for each segment: `4_brief_and_detailed_summarization.py`
