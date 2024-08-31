import os

import numpy as np
from label_studio_sdk.client import LabelStudio
from PIL import Image


LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY = '19563f95209292aa545aa0845388c18ad476f30c'
ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)


def create_project():
    pass

def get_project_by_name(project_name):
    projects = ls.projects.list()
    project_list = list(projects)
    number_of_projects = len(project_list)

    print(f'Number of projects: {number_of_projects}')
    for proj in project_list:
        if proj.title == project_name:
            return ls.projects.get(id=proj.id), id


def get_tasks(proj_id):
    return ls.tasks.list(project=1)

def create_tasks(proj_id, pkg_name, img_name):
    ls.tasks.create(
        project=proj_id,
        data={"image": f"data/local-files?d={pkg_name}/{img_name}", "text": f"{img_name}"}
    )

def update_task(proj_id, task_id, pkg_name, img_name):
    ls.tasks.update(
            id=task_id,
            data={"image": f"data/local-files?d={pkg_name}/{img_name}", "text": f"{img_name}"},
            project=proj_id,
        )

project, proj_id = get_project_by_name('Package 1')


# imgs_dir = r'C:\Users\fsalm\Desktop\projects\ULB\UGCT\UG_CT_images\imgs'
# labels_dir = r'C:\Users\fsalm\Desktop\projects\ULB\UGCT\UG_CT_images\labels'
# classes_file = r'C:\Users\fsalm\Desktop\projects\ULB\UGCT\UG_CT_images\classes.txt'
#
# im_name = '1.1_RCNX0012.JPG'
# root, _ = os.path.splitext(im_name)
# lbl_name = f'{root}.txt'
#
#
# print(im_name, lbl_name)
# img = np.array(Image.open(os.path.join(imgs_dir, im_name)))
# h,w,_ = img.shape
# data = []
# with open(os.path.join(labels_dir, lbl_name), 'r') as file:
#     for line in file:
#         # Split the line by spaces and convert each part to a float
#         numbers = list(map(float, line.split()))
#         # Append the list of numbers to the data list
#         data.append(numbers)
#
#
# classes = []
# with open(classes_file, 'r') as file:
#     for line in file:
#         classes.append(line.strip())
#
#
# results = []
# for item in data:
#     class_name = classes[int(item[0])]
#
#     width = item[3]
#     height = item[4]
#     x = item[1] - (width / 2)
#     y = item[2] - (height / 2)
#
#     x *= 100
#     y *= 100
#     width *= 100
#     height *= 100
#
#     results.append({
#                 "id": "unique_id_123",
#                 "type": "rectanglelabels",
#
#                 "value": {
#                 "x": x,
#                 "y": y,
#                 "width": width,
#                 "height": height,
#                 "rotation": 0,
#                 "rectanglelabels": [class_name]
#                 },
#                 "origin": "manual",
#                 "to_name": "image",
#                 "from_name": "label",
#
#                 "image_rotation": 0,
#                 "original_width": w,
#                 "original_height": h
#             })
#
# ls.annotations.create(
#     id=1,
#     result= results,
#     was_cancelled=False,
#     ground_truth=True,
# )