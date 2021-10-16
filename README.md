# MIIS-Capstone
This is the capstone project developed by the team "MIIS Big Four" at CMU SCS LTI

## Dependencies
Run the following command to configure the Python environment:

```shell script
conda create -n capstone python==3.8.0 -y
conda activate capstone
sudo apt-get update
sudo apt-get install python3-pil tesseract-ocr libtesseract-dev tesseract-ocr-eng -y
pip install -r requirements.txt
```

## Models
### Video Segmentation
Run the following commands:
```shell script
cd <PATH-TO-PROJECT>
export PYTHONPATH="$PWD"
python src/models/segmentation/video_segmentation.py --video-path <VIDEO_PATH>
```
where `<VIDEO_PATH>` is the path to the .mp4 file of the lecture