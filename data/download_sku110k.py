import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from ultralytics.utils.downloads import download
from ultralytics.utils.ops import xyxy2xywh

# Dataset path config
dir = Path("../datasets/SKU-110K")  # dataset root dir
parent = dir.parent  # parent folder
urls = ["http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"]

# Step 1: Download
download(urls, dir=parent)

# Step 2: Rename
if dir.exists():
    shutil.rmtree(dir)
(parent / "SKU110K_fixed").rename(dir)
(dir / "labels").mkdir(parents=True, exist_ok=True)

# Step 3: Convert CSV annotations to YOLO format
names = "image", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"
for d in ["annotations_train.csv", "annotations_val.csv", "annotations_test.csv"]:
    x = pd.read_csv(dir / "annotations" / d, names=names).values
    images, unique_images = x[:, 0], np.unique(x[:, 0])
    
    # Write txt list
    with open((dir / d).with_suffix(".txt").__str__().replace("annotations_", ""), "w", encoding="utf-8") as f:
        f.writelines(f"./images/{s}\n" for s in unique_images)
    
    # Write YOLO .txt labels
    for im in tqdm(unique_images, desc=f"Converting {dir / d}"):
        cls = 0  # single-class
        with open((dir / "labels" / Path(im).with_suffix(".txt")), "a", encoding="utf-8") as f:
            for r in x[images == im]:
                w, h = r[6], r[7]
                xywh = xyxy2xywh(np.array([[r[1] / w, r[2] / h, r[3] / w, r[4] / h]]))[0]
                f.write(f"{cls} {xywh[0]:.5f} {xywh[1]:.5f} {xywh[2]:.5f} {xywh[3]:.5f}\n")
