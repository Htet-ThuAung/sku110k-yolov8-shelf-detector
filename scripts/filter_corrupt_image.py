from PIL import Image
import os

# Path to your images
image_dir = r"D:\Projects\sku110k-yolov8-shelf-detector\datasets\SKU-110K\images"

bad = 0
for fname in os.listdir(image_dir):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    fpath = os.path.join(image_dir, fname)
    try:
        with Image.open(fpath) as im:
            im.verify()
    except Exception:
        print(f"Removing corrupt image: {fname}")
        os.remove(fpath)
        bad += 1

print(f"\n Done! Removed {bad} corrupt image(s).")


# D:\Projects\torch-gpu\Scripts\python.exe scripts\train.py
