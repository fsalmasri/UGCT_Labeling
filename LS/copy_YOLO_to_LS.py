import os
import pathlib
import numpy as np
from tqdm import tqdm
import json
from PIL import Image

from ls_ops import (
    get_project_by_name,
    fill_tasks)
from ls_ops import get_tasks, create_task, get_task_by_id, get_task_by_imName,generate_ls_uid, create_annot, update_annot
from ls_ops import fill_in_annotations

from LS.LS_utils import get_labels_from_file, get_classes



new_proj, new_proj_id = get_project_by_name('package 2')


img_dir = r'../../UGCT_images/2'
lbl_dir = r'../../UGCT_images/results/2'
classes_file = '../../DS/Package1-at-2024-09-13/classes.txt'

imgs_lst = os.listdir(img_dir)
lbl_files = os.listdir(lbl_dir)
classes = get_classes(classes_file)


def fill_annots():
    tasks = get_tasks(new_proj_id)
    for lbl_name in tqdm(lbl_files):
        root, _ = os.path.splitext(lbl_name)
        im = Image.open(os.path.join(img_dir, f'{root}.jpg'))
        width, height = im.size

        lbls = get_labels_from_file(lbl_dir, lbl_name)
        if len(lbls) > 0:
            results = fill_in_annotations(lbls, classes, width, height)

            task_id = get_task_by_imName(tasks, root).id
            create_annot(task_id=task_id, results=results)


# fill_tasks(folder_name='package2', proj_id=new_proj_id, imgs_lst=imgs_lst)
fill_annots()