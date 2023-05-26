from argparse import ArgumentParser, Namespace
from os.path import join
from typing import List, Tuple

import os

import cv2
import numpy as np
from pycocotools.coco import COCO

# (B,G,R) format.
# You may need to add more colors here
colors_dict = {
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'turquoise': (255, 255, 0),
    'white': (255, 255, 255),
    'yellow': (0, 255, 255),
    'purple': (255, 0, 255),
    'orange': (0, 128, 255),
    'brown': (0, 75, 150),
    'black': (0, 0, 0)
}


class Box:
    def __init__(self, bbox: Tuple[int, int, int, int], name: str, color: Tuple[int, int, int]):
        self.bbox = bbox
        self.name = name
        self.color = color


def plot_boxes(img: np.ndarray, boxes: List[Box]) -> np.ndarray:
    """
    Plots bounding boxes on the image
    :param img: image.
    :param boxes: bounding boxes.
    :return: image with bounding boxes plotted on it.
    """
    for box in boxes:
        x1 = int(box.bbox[0])
        y1 = int(box.bbox[1])
        x2 = int(box.bbox[0] + box.bbox[2])
        y2 = int(box.bbox[1] + box.bbox[3])
        img = cv2.putText(img, '%s' % (box.name), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, box.color, 2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), box.color, 3)
    return img


def make_video_COCO(args: Namespace):
    # Read COCO annotation
    ann_file_path = join(args.folder_path, 'lbl', 'COCO_annotation.json')
    coco = COCO(ann_file_path)
    anns = coco.loadAnns(coco.getAnnIds())

    tmp_img_id = anns[0]['image_id']
    bboxes = []
    image_name = coco.dataset['images'][tmp_img_id - 1]['file_name']

    image = cv2.imread(join(args.folder_path, image_name))
    for i, ann in enumerate(anns):
        if i == 0:
            height, width, _ = image.shape
            out = cv2.VideoWriter(filename=join(args.folder_path.split('/')[-1] + '_demo.mp4'),
                                  fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                  fps=args.fps,
                                  frameSize=(width, height))
        if ann['image_id'] != tmp_img_id:
            # Plot all bboxes for an image
            image = plot_boxes(img=image, boxes=bboxes)

            # Write image to out
            out.write(image)

            # Switch to next image
            image_name = coco.dataset['images'][ann['image_id'] - 1]['file_name']
            image = cv2.imread(join(args.folder_path, image_name))
            tmp_img_id = ann['image_id']
            bboxes = []
        class_name = coco.dataset['categories'][ann['category_id'] - 1]['name']

        # Collect all bboxes for an image
        box = Box(bbox=(ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3]),
                  name=class_name,
                  color=colors_dict[args.colors_list[args.classes_list.index(class_name)]])
        bboxes.append(box)
    out.release()


if __name__ == "__main__":
    arguments = ArgumentParser(description='Makes a video from images and labels')
    arguments.add_argument('-fp', '--folder_path', type=str, help='Path to folder with images and labels file')
    arguments.add_argument('-fps', '--fps', type=int, help='Out video FPS', default=25)
    arguments.add_argument('-clss', '--classes_list', nargs="+", type=str, help='Classes in a label file')
    arguments.add_argument('-clrs', '--colors_list', nargs="+", type=str, help='Colors for classes respectively')
    args = arguments.parse_args()

    make_video_COCO(args=args)

    print('Done!')
