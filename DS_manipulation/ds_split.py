import os

import random
import numpy as np
import pathlib
import shutil


from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from utils.DS_utils import get_classes, get_labels_from_file

ds_dir = '../../DS/Package1-at-2024-09-13'
imgs_dir =os.path.join(ds_dir, 'images')
lbls_dir = os.path.join(ds_dir, 'labels')

dir_to_save = '../../DS/Package1_training'
pathlib.Path(os.path.join(dir_to_save, 'train', 'images')).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.join(dir_to_save, 'train', 'labels')).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.join(dir_to_save, 'validation', 'images')).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.join(dir_to_save, 'validation', 'labels')).mkdir(parents=True, exist_ok=True)


imgs_list = os.listdir(imgs_dir)
classes = get_classes(os.path.join(ds_dir, 'classes.txt'))

classes_dic = {}
classes_names_dic = {}
empty_images = []
for im_name in imgs_list:
    im = Image.open(os.path.join(imgs_dir, im_name))
    width, height = im.size

    root, _ = os.path.splitext(im_name)
    lbl_name = f'{root}.txt'

    lbls = get_labels_from_file(lbls_dir, lbl_name)
    if lbls is not None:
        if len(lbls) > 0:
            class_names = [classes[int(x[0])] for x in lbls]
            areas = [((x[3]*width)*(x[4]*height))/(width*height) for x in lbls]

            for class_name, area in zip(class_names, areas):
                if class_name not in classes_dic:
                    classes_dic[class_name] = 1
                    classes_names_dic[class_name] = [im_name]
                else:
                    classes_dic[class_name] += 1
                    classes_names_dic[class_name].append(im_name)
        else:
            empty_images.append(im_name)


print(classes_dic)
cls_percent = {k: int(v*.1) for k,v in classes_dic.items()}

print(cls_percent)
print(np.sum(list(classes_dic.values())), np.sum(list(cls_percent.values())))


validation_list = []
training_list = []
for k, v in classes_names_dic.items():
    num_samples = int(len(v) * 0.15)

    random_items = random.sample(v, num_samples)
    remaining_items = [x for x in v if x not in random_items]

    validation_list.extend(random_items)
    training_list.extend(remaining_items)

validation_list = np.unique(validation_list)
training_list = np.unique(empty_images + training_list)
training_list = [x for x in training_list if x not in validation_list]


for im_name in training_list:
    root, _ = os.path.splitext(im_name)
    lbl_name = f'{root}.txt'

    shutil.copy(os.path.join(imgs_dir, im_name), os.path.join(dir_to_save, 'train', 'images', im_name))
    shutil.copy(os.path.join(lbls_dir, lbl_name), os.path.join(dir_to_save, 'train', 'labels', lbl_name))

for im_name in validation_list:
    root, _ = os.path.splitext(im_name)
    lbl_name = f'{root}.txt'

    shutil.copy(os.path.join(imgs_dir, im_name), os.path.join(dir_to_save, 'validation', 'images', im_name))
    shutil.copy(os.path.join(lbls_dir, lbl_name), os.path.join(dir_to_save, 'validation', 'labels', lbl_name))

