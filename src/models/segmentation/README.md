## Dependencies
Run the following command to configure the Python environment:

```shell script
conda create -n capstone python==3.8.0 -y
conda activate capstone
sudo apt-get update
sudo apt-get install python3-pil tesseract-ocr libtesseract-dev tesseract-ocr-eng -y
pip install -r requirements.txt
```

## How To?
Run the following commands:
```shell script
cd <PATH-TO-PROJECT>
export PYTHONPATH="$PWD"
python src/models/segmentation/video_segmentation.py --video-path <VIDEO_PATH>
```
where `<VIDEO_PATH>` is the path to the .mp4 file of the lecture


## API Design
Function: `video_segmentation(opt: argparse.Namespace) -> List[str]` in `video_segmentation.py`

|        | Type               | Description                                   |
|--------|--------------------|-----------------------------------------------|
| Input  | argparse.Namespace | Arguments (see video_segmentation.py)         |
| Output | List[str]          | List of timestamps of boundaries ("%H:%M:%S") |

Please see 'main' function in `video_segmentation.py` for reference.
