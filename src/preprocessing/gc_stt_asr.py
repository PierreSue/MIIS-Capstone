# encoding: utf-8
"""
@author: Jingrong Feng
@contact: jingronf@andrew.cmu.edu
@version: 0.1
@file: gc_stt_asr.py
@time: 10/14/21

Prerequisites: Follow step2 to step5 on setup environment and authenticate API requests
"""
from time import time
from typing import List

from google.cloud import speech

from src.preprocessing.utils import START_TIMESTAMP


def get_word_offsets(alternative) -> List[str]:
    timestamp_word_lines = []
    for word in alternative.words:
        start_s = (word.start_time + START_TIMESTAMP).isoformat(sep=' ', timespec='milliseconds').split(' ')[1]
        end_s = (word.end_time + START_TIMESTAMP).isoformat(sep=' ', timespec='milliseconds').split(' ')[1]
        word = word.word
        timestamp_word_lines.append(f"{start_s} | {end_s} | {word}")
    return timestamp_word_lines


def transcribe_gcs(gcs_uri: str, output_file: str) -> List[str]:
    """
    Asynchronously transcribes the audio file specified by the gcs_uri.
    cf. https://cloud.google.com/speech-to-text/docs/async-recognize#speech_transcribe_async_gcs-python
    and https://codelabs.developers.google.com/codelabs/cloud-speech-text-python3#7
    """

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=32000,
        language_code="en-US",
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        model="video",
        use_enhanced=True
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    tic = time()
    response = operation.result(timeout=10000)
    toc = time()
    print(f"===== {toc - tic}s =====")

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    timestamp_word_lines = []
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(u"Transcript: {}".format(result.alternatives[0].transcript))
        print("Confidence: {}".format(result.alternatives[0].confidence))
        timestamp_word_lines.extend(get_word_offsets(result.alternatives[0]))

    with open(output_file, "w") as txt_file:
        txt_file.write("\n".join(timestamp_word_lines))
    print(f"Transcript with timestamps has been saved to {output_file}")


def run():
    gc_uri = "gs://miis-capstone-lvs-asr/Fall2020-SpeechProcessing/2020_09_02_11_492_02_Human_Speech.wav"
    output_file = "/home/jingrong/capstone/data/gc_stt/2020_09_02_11_492_02_Human_Speech_transcript.txt"
    transcribe_gcs(gc_uri, output_file)


if __name__ == '__main__':
    run()
