import os
import argparse
from PIL import Image
from models.segmentation import video_segmentation


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Video Segmentation
    parser.add_argument('--video-path', type=str, help='The input video path')
    parser.add_argument('--output-path', type=str, help='The output json file path')
    parser.add_argument('--image-dir', type=str, help='The output image directory')
    parser.add_argument('--interval', type=int, default=2,
                        help='The interval between each image in seconds. (An image/N sec)')
    parser.add_argument('--threshold-pixel', type=int, default=0.05,
                        help='The threshold of pixel differences (1st stage)')
    parser.add_argument('--boundaries', type=str, default='105,120,675,885',
                        help='The boundaries of slides: x_topleft, y_topleft, x_bottomright, y_bottomright')
    
    parser.add_argument('--edit-distance-threshold', type=float, default=0.6,
                        help='The maximum (Levenshtein_distance(s_{i}, s_{i+1}) / s_{i+1}) '
                             'between two slides s_{i} and s_{i+1} within a segment (2nd stage)')
    parser.add_argument('--min-interval', type=int, default=20,
                        help='The minimum time interval of a segment (in seconds)')
    parser.add_argument('--num-frame-forward', type=int, default=10,
                        help='The number of frames we look forward to determine the boundaries (2nd stage)')
    parser.add_argument('--bi-threshold', type=int, default=100,
                        help='The threshold for image binarization (0, 255), 0->black and 1->white (2nd stage)')

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_arguments()

    # mkdirs
    if not os.path.exists(os.path.dirname(opt.output_path)):
        os.makedirs(os.path.dirname(opt.output_path))
    if not os.path.exists(opt.image_dir):
        os.makedirs(opt.image_dir)

    valid_boundaries, segments, images = video_segmentation(opt)
    for i, (segmentA, segmentB) in enumerate(zip(segments[:-1], segments[1:])):
        im = Image.fromarray(images[(segmentA+segmentB)//2])
        im.save(os.path.join(opt.image_dir, '{}.jpg'.format(i)))