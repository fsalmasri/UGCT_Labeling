import os
import pandas as pd
from pathlib import Path



ds_dir = '../../UG_CT_images'
df = pd.read_csv(os.path.join(ds_dir, 'updated_species_annotations.csv'))

df.columns = df.columns.str.strip()

# Delete no animal class
df_filtered = df[df['label'] != 'no animal']

# collect labels and re-order classes sequentially.
unique_classes = sorted(df_filtered['classification'].unique())
class_mapping = {x: idx for  idx, x in enumerate(unique_classes[1:], start=1)}
class_mapping[0] = 0

# map classes and save new file.
df_filtered.loc[:, 'classification'] = df_filtered['classification'].map(class_mapping)
# df_filtered.to_csv(os.path.join(ds_dir, 'mapped_data.csv'), index=False)

# extract labels and save classes.txt file
grouped = df_filtered.groupby('classification')['label'].unique()
labels = [label for labels in grouped for label in labels]
classes_file = os.path.join(ds_dir, "classes.txt")
# with open(classes_file, "w") as file:
#     for item in labels:
#         file.write(f"{item}\n")


# Create labels folder
Path(os.path.join(ds_dir, 'labels')).mkdir(parents=True, exist_ok=True)


image_dict = {}
# Loop and save annotations in YOLO format.
for index, row in df_filtered.iterrows():
    basename = os.path.basename(row['image_path'])
    row_values = [
        row['classification'],
        row['x_center'],
        row['y_center'],
        row['width'],
        row['height']
    ]

    if basename not in image_dict:
        image_dict[basename] = []

    image_dict[basename].append(row_values)

for key, value in image_dict.items():
    root, _ = os.path.splitext(key)
    file_name = f'{root}.txt'

    label_file = os.path.join(ds_dir, "labels", file_name)
    with open(label_file, "w") as file:
        for item in value:
            line = ' '.join(map(str, item))
            file.write(f"{line}\n")



