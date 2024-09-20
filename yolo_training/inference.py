import os

import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics import settings

import torch
from PIL import Image


def inference():
    model_name = 'yolov8n_custom2'

    model = YOLO(f'runs/detect/{model_name}/weights/best.pt', task='detect')

    im_dir = r'../../UGCT_images/2'
    im_lst = os.listdir(im_dir)
    print(f'Images count: {len(im_lst)}')

    saving_dir = '../../UGCT_images/results/2'
    os.makedirs(saving_dir, exist_ok=True)

    for im in im_lst:
        results = model(os.path.join(im_dir, im))
        for i, result in enumerate(results):
            boxes = result.boxes.xywhn  # Get xywh format
            classes = result.boxes.cls  # Get classes

            save_path = os.path.join(saving_dir, f'{os.path.splitext(im)[0]}.txt')
            with open(save_path, 'w') as f:
                for box, cls in zip(boxes, classes):
                    print(box, cls)
                    f.write(f'{int(cls)} {box[0]} {box[1]} {box[2]} {box[3]}\n')



if __name__ == '__main__':
    inference()