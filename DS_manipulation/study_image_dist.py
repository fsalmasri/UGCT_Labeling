import os
# from tkinter import Image

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import imutils

from utils.DS_utils import get_classes, get_labels_from_file


def image_colorfulness(image):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)

ds_dir = '../../UGCT/DS/package_1_2'
imgs_dir =os.path.join(ds_dir, 'images')
lbls_dir = os.path.join(ds_dir, 'labels')

imgs_list = os.listdir(imgs_dir)
classes = get_classes(os.path.join(ds_dir, 'classes.txt'))

empty_images =0
classes_dic = {}
normalized_area_dist = {}
color = []

def get_color_dist():
    for im_name in tqdm(imgs_list):
        im = cv2.imread(os.path.join(imgs_dir, im_name))
        width, height, _ = im.shape

        im = imutils.resize(im, width=250)
        colorfulness = image_colorfulness(im)
        color.append(colorfulness)
        if colorfulness < 2.5:
            plt.imshow(im)
            plt.show()
        # root, _ = os.path.splitext(im_name)
        # lbl_name = f'{root}.txt'
        #
        # lbls = get_labels_from_file(lbls_dir, lbl_name)
        # if lbls is not None and len(lbls) > 0:
        #     pass
        # else:
        #     empty_images += 1

    np.save('color.npy', np.array(color))

# mean_norm_area_dist = {k:np.mean(v) for k, v in normalized_area_dist.items()}

# get_color_dist()

color = np.load('color.npy')
gray = [x for x in color if x < 2.5]
colored = [x for x in color if x >= 2.5]

print(f'gray images {100*len(gray)/len(color):.2f}%     colored images {100*len(colored)/len(color):.2f}% ')

# print(f'{empty_images} empty images of {len(imgs_list)} images; {100*empty_images/len(imgs_list):0.2f}%')


plt.hist(color, bins=100)
plt.show()
