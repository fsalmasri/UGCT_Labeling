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



def draw_boxes(image, boxes, classes, confidences, class_names, thickness=2):
    COLOR_PALETTE = [
        (255, 0, 0),  # Class 0: Red
        (0, 255, 0),  # Class 1: Green
        (0, 0, 255),  # Class 2: Blue
        # (255, 255, 0),  # Class 3: Cyan
        (255, 0, 255),  # Class 4: Magenta
        # (0, 255, 255),  # Class 5: Yellow
    ]

    import cv2

    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        color = COLOR_PALETTE[int(cls) % len(COLOR_PALETTE)]
        label = f"{class_names[int(cls)]} {conf:.2f}"

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Put the class label
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
        text_x, text_y = x1, y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), color,
                      -1)  # Text background
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness)

    return image


def save_prediction(filepath, blocks):
    with open(filepath, 'w') as f:
        for _list in blocks:
            # print(_list)
            for _string in _list:
                f.write(str(_string) + ' ')
            f.write('\n')

def test(no):
    model = YOLO(f'runs/detect/{no}/weights/best.pt', task='detect')
    test_dir = f'../DS/LS_Will_DS/v2/valid/images'


    for test_im in os.listdir(test_dir)[:15]:

        im = np.array(Image.open(os.path.join(test_dir, test_im)))
        # im = im[:1500, :1500]
        # print(im.shape)
        im = np.stack((im, im, im), axis=2)
        # print(im.shape)
        # exit()
        # plt.imshow(im)
        # plt.show()
        # exit()
        # results = model.predict(os.path.join(test_dir, test_im), save=True, line_width=1, conf=.1)
        results = model.predict(im, save=False, line_width=1, conf=.1, imgsz=(im.shape[1], im.shape[0]),  iou=0.3)
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        names = result.names


        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        classes = results[0].boxes.cls.cpu().numpy()  # Class IDs
        confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        class_names = model.names  # Class names (YOLO model provides these)

        blocks = [[x[0], x[1], x[2], x[3], class_names[y], z] for x, y, z in zip(boxes, classes, confidences) if class_names[y] == 'TEXT']

        # SAVE prediction in file.
        # save_prediction(f'{filepath}/{test_im[:-3]}txt', blocks)

        # Draw boxes with class-specific colors
        annotated_image = draw_boxes(im.copy(), boxes, classes, confidences, class_names)



        filepath = Path(f'runs/predict/{no}')
        filepath.mkdir(parents=True, exist_ok=True)
        Image.fromarray(annotated_image).save(f'{filepath}/{test_im}')



        # annotated_image = results[0].plot(font_size=0)

        # plt.imshow(annotated_image)
        # plt.show()

        # exit()

        # predicted_LC_bbxs = boxes[classes == 0]
        # predicted_LC_conf = confidences[classes == 0]
        #
        # predicted_LCCON_bbxs = boxes[classes == 1]
        # predicted_LCCON_conf = confidences[classes == 1]
        #
        # predicted_LCInp_bbxs = boxes[classes == 2]
        # predicted_LCInp_conf = confidences[classes == 2]
        #
        # predicted_txt_bbxs = boxes[classes == 3]
        # predicted_txt_conf = confidences[classes == 3]
        #
        # print(len(classes), len(predicted_LC_bbxs), len(predicted_LCCON_bbxs))
        #
        # np.save('LC_bbx.npy', predicted_LC_bbxs)
        # np.save('LCCON_bbx.npy', predicted_LCCON_bbxs)
        # np.save('LCInp_bbx.npy', predicted_LCInp_bbxs)
        # np.save('LCTXT_bbx.npy', predicted_txt_bbxs)
        # exit()
        #     result.save(show_labels=False)


def validate_model(no):
    model = YOLO(f'runs/detect/{no}/weights/best.pt', task='detect')
    model.val(data='ds.yaml', batch=1, save_json=True)



if __name__ == '__main__':
    exp = 2
    # run(2)
    test()