import os
# from tkinter import Image

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import imutils
import random
import shutil


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
bg_dir = os.path.join(ds_dir, 'BG')

imgs_list = os.listdir(imgs_dir)
classes = get_classes(os.path.join(ds_dir, 'classes.txt'))


def clear_bg_images():
    empty_images = 0

    dic = {'0': [], '1': []}
    for im_name in tqdm(imgs_list):
        im = cv2.imread(os.path.join(imgs_dir, im_name))
        width, height, _ = im.shape
        root, _ = os.path.splitext(im_name)
        lbl_name = f'{root}.txt'

        lbls = get_labels_from_file(lbls_dir, lbl_name)
        if lbls is None or len(lbls) == 0:
            im = imutils.resize(im, width=250)
            colorfulness = image_colorfulness(im)

            if colorfulness < 2.5:
                dic['0'].append(im_name)
            else:
                dic['1'].append(im_name)


    np.save('gray_list.npy', dic['0'])
    np.save('color_list.npy', dic['1'])

# clear_bg_images()


graylst = np.load('gray_list.npy')
colorlst = np.load('color_list.npy')


# no_fg_images = len(imgs_list) - len(graylst) - len(colorlst)
# ds_size = no_fg_images /.9
# bg_to_keep = ds_size - no_fg_images
# bg_ro_remove = (len(graylst)+len(colorlst))-bg_to_keep
#
#
# print(f'Gray: {len(graylst)}, {100 * len(graylst) / len(imgs_list):.2f}%, Color: {len(colorlst)}, {100 * len(colorlst) / len(imgs_list):.2f}%, Total image: {len(imgs_list)}')
# print(f'no of FG: {no_fg_images}, no of BG to keep: {bg_to_keep}, no of BG to remove {bg_ro_remove:.2f}')
#
# print(f'gray to keep:{bg_to_keep*0.1358:.2f}, color to keep:{bg_to_keep*0.3785:.2f}')

random.shuffle(graylst)

gray_to_remove = graylst[63:]
color_to_remove = colorlst[174:]


# for g in gray_to_remove:
#     shutil.move(os.path.join(imgs_dir, g), os.path.join(bg_dir, g))
#
#     lbl_name = f'{g[:-3]}txt'
#     lbl_path = os.path.join(lbls_dir, lbl_name)
#     if os.path.exists(lbl_path):
#         shutil.move(lbl_path, os.path.join(bg_dir, lbl_name))
#
#
# for g in color_to_remove:
#     shutil.move(os.path.join(imgs_dir, g), os.path.join(bg_dir, g))
#
#     lbl_name = f'{g[:-3]}txt'
#     lbl_path = os.path.join(lbls_dir, lbl_name)
#     if os.path.exists(lbl_path):
#         shutil.move(lbl_path, os.path.join(bg_dir, lbl_name))
#



