## Dependencies
Run the following command to configure the Python environment:

```shell script
git clone git@github.com:Yale-LILY/SummerTime.git
cd SummerTime
conda create -n summertime python=3.7 -y
conda activate summertime
pip install -e .
```

## API Design
Function: `detailed_summary(batch_paragraph: List[str]) -> List[str]` in `bart_inference.py`

|        | Type       | Description                                                  |
|--------|------------|--------------------------------------------------------------|
| Input  | List[str]  | List of transcript segments                                  |
| Output | List[str]  | List of summaries generated using BART, one for each segment |

Please see `test_bart_inference.py` for reference.
