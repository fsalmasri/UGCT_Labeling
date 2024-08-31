import os

import numpy as np
from label_studio_sdk.client import LabelStudio
from PIL import Image
import uuid
import secrets
import string
from tqdm import tqdm


LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY = '19563f95209292aa545aa0845388c18ad476f30c'
ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

imgs_dir = r'C:\Users\fsalm\Desktop\projects\ULB\UGCT\UG_CT_images\imgs'
labels_dir = r'C:\Users\fsalm\Desktop\projects\ULB\UGCT\UG_CT_images\labels'
classes_file = r'C:\Users\fsalm\Desktop\projects\ULB\UGCT\UG_CT_images\classes.txt'

def create_project():
    pass

def get_project_by_name(project_name):
    projects = ls.projects.list()
    project_list = list(projects)
    number_of_projects = len(project_list)

    print(f'Number of projects: {number_of_projects}')
    for proj in project_list:
        if proj.title == project_name:
            return ls.projects.get(id=proj.id), proj.id


def get_tasks(proj_id):
    tasks = ls.tasks.list(project=proj_id)
    tasks_list = list(tasks)
    number_of_tasks = len(tasks_list)

    print(f'Number of tasks: {number_of_tasks}')

    return tasks

def create_task(proj_id, folder_name, img_name):
    ls.tasks.create(
        project=proj_id,
        data={"image": f"data/local-files?d={folder_name}/{img_name}", "text": f"{img_name}"}
    )

def update_task(proj_id, task_id, pkg_name, img_name):
    ls.tasks.update(
            id=task_id,
            data={"image": f"data/local-files?d={pkg_name}/{img_name}", "text": f"{img_name}"},
            project=proj_id,
        )

project, proj_id = get_project_by_name('Package 1')


def add_tasks_from_folder(proj_id, imgs_dir, folder_name):
    flst = os.listdir(imgs_dir)

    for f in flst:
        create_task(proj_id, folder_name, img_name=f)



def get_classes(classes_file):
    classes = []
    with open(classes_file, 'r') as file:
        for line in file:
            classes.append(line.strip())

    return classes

def get_labels_from_file(labels_dir, lbl_name):
    data = []
    try:
        with open(os.path.join(labels_dir, lbl_name), 'r') as file:
            for line in file:
                # Split the line by spaces and convert each part to a float
                numbers = list(map(float, line.split()))
                # Append the list of numbers to the data list
                data.append(numbers)
        return data
    except:
        return None

def generate_ls_uid(length=10):
    first_char = secrets.choice(string.ascii_lowercase)
    other_chars = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))

    return f'{first_char}_{other_chars}'

def fill_in_annotations(data, classes, im_w, im_h):
    results = []
    for item in data:
        class_name = classes[int(item[0])]

        width = item[3]
        height = item[4]
        x = item[1] - (width / 2)
        y = item[2] - (height / 2)

        x *= 100
        y *= 100
        width *= 100
        height *= 100

        results.append({
                    "id": f"{generate_ls_uid()}",
                    "type": "rectanglelabels",

                    "value": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rotation": 0,
                    "rectanglelabels": [class_name]
                    },
                    "origin": "manual",
                    "to_name": "image",
                    "from_name": "label",

                    "image_rotation": 0,
                    "original_width": im_w,
                    "original_height": im_h
                })

    return results

def add_annotations_from_tasks(proj_id):
    classes = get_classes(classes_file)

    tasks = get_tasks(proj_id)
    for task in tqdm(tasks):
        task_id = task.id
        im_name = task.data['text']

        root, _ = os.path.splitext(im_name)
        lbl_name = f'{root}.txt'

        img = np.array(Image.open(os.path.join(imgs_dir, im_name)))
        im_h, im_w, _ = img.shape

        data = get_labels_from_file(labels_dir, lbl_name)
        if data:
            results = fill_in_annotations(data, classes, im_w, im_h)
        else:
            results = []
        ls.annotations.create(
            id=task_id,
            result= results,
            was_cancelled=False,
            ground_truth=True,
        )

# add_tasks_from_folder(proj_id, imgs_dir, 'package1')

# add_annotations_from_tasks(proj_id)
