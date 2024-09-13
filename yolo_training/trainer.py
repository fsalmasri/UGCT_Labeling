import os

import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics import settings

import torch
import fitz
from PIL import Image



def run():
    model_name = 'yolov8n_custom1'

    settings.update({'runs_dir': '/runs',
                     'tensorboard': True,
                     'visualize': True
                     })

    # Load the model.
    model = YOLO('yolov8n.pt', task='detect')
    # model = YOLO(
    #     f'runs/detect/{model_name}/weights/last.pt',
    #              task='detect')

    # Training.
    results = model.train(
        data='ds.yaml',
        imgsz=1280,
        epochs=100,
        batch=14,
        workers=6,

        pose=0,
        kobj=0,

        dropout=0.3,
        scale= 0.5,
        fliplr= False,
        show_labels=False,

        name=f'{model_name}',
        # resume=True
        save=True

    )



def test():
    model_name = 'yolov8n_custom16'

    model = YOLO(f'runs/detect/{model_name}/weights/best.pt', task='detect')

    im_dir = r'..\DS\project-2-at-2024-08-31-21-30-9814d192-cropped\valid\images'
    im_lst = os.listdir(im_dir)

    for im in im_lst:
        results = model(os.path.join(im_dir, im))
        for i, result in enumerate(results):
            result.save(labels=True, line_width=1, font_size=3)  # or .show()



if __name__ == '__main__':
    run()
    # test()