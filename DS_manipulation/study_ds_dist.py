import os
from tkinter import Image

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from utils.DS_utils import get_classes, get_labels_from_file

ds_dir = '../../DS/Package1-at-2024-09-13'
imgs_dir =os.path.join(ds_dir, 'images')
lbls_dir = os.path.join(ds_dir, 'labels')

imgs_list = os.listdir(imgs_dir)
classes = get_classes(os.path.join(ds_dir, 'classes.txt'))

classes_dic = {}
normalized_area_dist = {}
for im_name in imgs_list:
    im = Image.open(os.path.join(imgs_dir, im_name))
    width, height = im.size

    root, _ = os.path.splitext(im_name)
    lbl_name = f'{root}.txt'

    lbls = get_labels_from_file(lbls_dir, lbl_name)
    if lbls is not None and len(lbls) > 0:
        class_names = [classes[int(x[0])] for x in lbls]
        areas = [((x[3]*width)*(x[4]*height))/(width*height) for x in lbls]

        for class_name, area in zip(class_names, areas):
            if class_name not in classes_dic:
                classes_dic[class_name] = 1
                normalized_area_dist[class_name] = [area]
            else:
                classes_dic[class_name] += 1
                normalized_area_dist[class_name].append(area)

mean_norm_area_dist = {k:np.mean(v) for k, v in normalized_area_dist.items()}


# to_plot = classes_dic
to_plot = mean_norm_area_dist

names = list(to_plot.keys())
values = list(to_plot.values())

plt.figure(figsize=(10,5))
plt.bar(names, values, color='skyblue')
plt.xlabel('Species')
plt.ylabel('Count')
plt.title('Species vs Count')
plt.xticks(rotation=45, ha='right')

# Display the plot
plt.tight_layout()

plt.show()