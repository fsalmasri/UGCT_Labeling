import os

import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics import settings

import torch
from PIL import Image



def run(no):
    # model_name = 'yolov11n_custom1'
    saving_model_name = f'{no}'

    settings.update({'runs_dir': 'runs',
                     'tensorboard': True,
                     'datasets_dir': '.'
                     # 'visualize': True
                     })

    # Load the model.
    model = YOLO('yolo11n.pt', task='detect')
    # model = YOLO(
    #     f'runs/detect/{model_name}/weights/last.pt',
    #              task='detect')

    # Training.
    results = model.train(
        data='ds.yaml',
        imgsz=1024,
        epochs=500,
        batch=32,
        workers=12,
        device=[0, 1],

        pose=0,
        kobj=0,
        patience=100,

        lr0= 0.001,

        dropout=0.3,
        scale= 0.3,
        hsv_h= 0.2,
        hsv_s= 0.4,
        hsv_v= 0.4,
        fliplr= False,

        show_labels=False,

        name=f'{saving_model_name}',
        # resume=True,
        save=True,

        line_width=1

    )



def test():
    model_name = 'yolov8n_custom18'

    model = YOLO(f'runs/detect/{model_name}/weights/best.pt', task='detect')

    im_dir = r'../../UGCT/DS/package_1_2/validation/images'
    im_lst = os.listdir(im_dir)

    for im in im_lst:
        results = model(os.path.join(im_dir, im))
        for i, result in enumerate(results):
            result.save(labels=True, line_width=1, font_size=3, filename=f'runs/predict/{im}')  # or .show()



if __name__ == '__main__':
    exp = 2
    # run(2)
    test()