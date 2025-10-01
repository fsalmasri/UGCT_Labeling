import os.path
import pandas as pd
import json
import os
import pandas as pd
import re

ds = r'C:\Users\fsalm\Desktop\Projects\dataset\ULB\wildlife-insights_ff7cb184-86f7-4ff3-8107-5315224e997a_project-2009350_data'
csv_path = "images_2009350.csv"
df = pd.read_csv(os.path.join(ds, csv_path), low_memory=False)


s = df["bounding_boxes"].astype(str).str.strip().str.lower()
mask = df["bounding_boxes"].notna() & s.ne("") & ~s.isin({"nan","none","null","[]","{}"})

print("total no of images:", len(df))
print("with_bbx:", int(mask.sum()))
print("without_bbx:", int((~mask).sum()))



# --- Parse detection boxes from strings like: {"\"{...}\"","\"{...}\""}
pat = re.compile(r'detectionBox\\?\"\s*:\s*\[([^\]]+)\]')  # handles \"detectionBox\" and detectionBox"

def parse_boxes(s):
    if pd.isna(s):
        return []
    s = str(s)
    nums = []
    for m in pat.findall(s):
        try:
            nums.append([float(x) for x in m.split(",")])
        except Exception:
            pass
    return nums

# Parse and keep only rows with >=1 box
df["boxes"] = df["bounding_boxes"].map(parse_boxes)
with_boxes = df[df["boxes"].str.len() > 0].copy()

# Keep attributes you care about (add/remove as needed)
attrs = [
    "project_id","deployment_id","image_id","filename","location",
    "number_of_objects",
    "class","order","family","genus","species","common_name","boxes"
]
keep_cols = [c for c in attrs if c in with_boxes.columns]
result = with_boxes[keep_cols]

print(result.head(100).to_string(index=False))
exit()

# Optional: one row per box (explode)
per_box = result.explode("boxes", ignore_index=True)

# Save/print
out1 = os.path.join(ds, "images_with_bbx.csv")
out2 = os.path.join(ds, "images_with_bbx_perbox.csv")
result.to_csv(out1, index=False)
per_box.to_csv(out2, index=False)

print(f"Kept {len(result)} images with boxes.")
print(per_box.head(100).to_string(index=False))


# Count boxes per image
df["boxes_count"] = df["boxes"].str.len()

print("total images:", len(df))
print("with_bbx:", int((df["boxes_count"] > 0).sum()))
print("without_bbx:", int((df["boxes_count"] == 0).sum()))
print("total boxes:", int(df["boxes_count"].sum()))
print("\nboxes per image (distribution):")
print(df["boxes_count"].value_counts().sort_index().to_string())

# peek at a few with boxes
print("\nSample with boxes:")
print(df.loc[df["boxes_count"] > 0, ["filename", "location", "boxes_count"]].head(10).to_string(index=False))



# use the filtered df: result = with_boxes[keep_cols]
col = next((c for c in result.columns if c.lower() == "common_name"), None)
if col is None:
    raise KeyError("Column 'common_name' not found in result")

labels = result[col].astype(str).str.strip()
valid = labels.ne("") & ~labels.str.lower().isin({"blank","nan","none","null"})

# unique class list (among images with boxes)
classes = sorted(labels[valid].unique(), key=str.casefold)
print("Classes (with boxes):", len(classes), classes)

# counts per class (with boxes)
print("\nCounts (with boxes):")
print(labels[valid].value_counts().to_string())

# assuming `classes` is already computed
out_path = os.path.join(ds, "classes_with_boxes.txt")  # or just "classes.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(map(str, classes)))
print(f"Saved {len(classes)} classes to: {out_path}")


# known classes (baseline list)
known_raw = """
alcelaphus buselaphus
canis adustus
crocuta crocuta
erythrocebus patas
fire
genetta tigrina
giraffa camelopardalis
hippopotamus amphibius
kobus ellipsiprymnus
kobus kob
loxodonta africana
orycteropus afer
ourebia ourebi
panthera leo
papio anubis
pathera pardus
person
phacochoerus africanus
syncerus caffer
unknown animal
vehicle
""".strip().splitlines()

print(known_raw)


norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
known_set = {norm(x) for x in known_raw}
orig_by_norm = {norm(c): c for c in classes}


# find new classes (by normalized form), keep original spellings from `classes`
new_norm = sorted(set(orig_by_norm.keys()) - known_set)
new_classes = [orig_by_norm[n] for n in new_norm]


# write merged list (keep known order, then new ones)
out_path = os.path.join(ds, "class_list.txt")
merged = known_raw[:] + [orig_by_norm[n] for n in new_norm if n not in known_set]

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(merged))

print(f"New classes ({len(new_classes)}):")
print("\n".join(new_classes) if new_classes else "None")
print(f"Updated class list saved to: {out_path}")