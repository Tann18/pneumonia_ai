import os
import pandas as pd
import shutil

# -----------------------------
# PATHS (MATCH YOUR STRUCTURE)
# -----------------------------
CSV_PATH = "labels_train.csv"
IMAGES_DIR = "train_images/train_images"
OUTPUT_DIR = "data/train"

# -----------------------------
# CLASS MAPPING
# -----------------------------
CLASS_MAP = {
    0: "NORMAL",
    1: "BACTERIAL",
    2: "VIRAL"
}

# -----------------------------
# CREATE OUTPUT FOLDERS
# -----------------------------
for folder in CLASS_MAP.values():
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

# -----------------------------
# READ CSV
# -----------------------------
df = pd.read_csv(CSV_PATH)

# -----------------------------
# COPY IMAGES
# -----------------------------
missing = 0
copied = 0

for _, row in df.iterrows():
    img_name = row["file_name"]
    class_id = row["class_id"]

    src = os.path.join(IMAGES_DIR, img_name)
    dst = os.path.join(OUTPUT_DIR, CLASS_MAP[class_id], img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)
        copied += 1
    else:
        missing += 1

print("✅ DONE")
print("Images copied:", copied)
print("Missing images:", missing)