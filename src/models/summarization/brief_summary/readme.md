# Documentation

## Pipeline explanation

1. Read input texts from given path, which are already grammarly restored and checked.
2. Pass to Pegasus model.
3. Write result into output json file.

## Input and output format


Input is a json array, in which each element is a dictionary containing timestamp and restored trascript. 

[
    
    {
        "start_timestamp": "00:00:00",
        "transcript": "Okay, xxxx vocal",
        "transcript_corrected": "Okay, xxxx vocal"
    }
]

I will add an abstractive summarization into the output file, which makes each element look like this:

[

    {
        "start_timestamp": "00:00:00",
        "transcript": "Okay, xxxx vocal",
        "transcript_corrected": "Okay, xxxx vocal",
        "summary_brief": "xx"
    }

]

Finetuned model will be automatically downloaded if not existed. Remember to set the PYTHONPATH to the root directory of the project.