import os, sys
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

api = HfApi()
files = list(api.list_repo_files("PranomVignesh/MRI-Images-of-Brain-Tumor", repo_type="dataset"))
print(f"Total files in repo: {len(files)}")

image_files = [f for f in files if f.endswith(".jpg")]
print(f"Image files: {len(image_files)}")

import random
random.seed(42)

def download_subset(split_name, max_per_class=30):
    split_files = [f for f in image_files if f.startswith(f"timri/{split_name}/")]
    print(f"\n{split_name}: {len(split_files)} images available")
    classes = {}
    for f in split_files:
        parts = f.split("/")
        if len(parts) >= 3:
            cls = parts[2]
            if cls not in classes:
                classes[cls] = []
            classes[cls].append(f)
    print(f"  Classes: {list(classes.keys())}")
    for cls, cls_files in classes.items():
        target_dir = os.path.join(DATA_DIR, split_name, cls)
        os.makedirs(target_dir, exist_ok=True)
        selected = random.sample(cls_files, min(max_per_class, len(cls_files)))
        for f in tqdm(selected, desc=f"  Downloading {cls}"):
            dest = os.path.join(target_dir, os.path.basename(f))
            if not os.path.exists(dest):
                hf_hub_download(
                    "PranomVignesh/MRI-Images-of-Brain-Tumor",
                    f, repo_type="dataset",
                    local_dir=DATA_DIR,
                    local_dir_use_symlinks=False,
                )
        # move file to correct location if hf_hub puts it in subdir
        src_base = os.path.join(DATA_DIR, "timri", split_name, cls)
        if os.path.exists(src_base):
            for src_file in os.listdir(src_base):
                src_path = os.path.join(src_base, src_file)
                dst_path = os.path.join(target_dir, src_file)
                if not os.path.exists(dst_path):
                    os.rename(src_path, dst_path)
        print(f"  Downloaded {len(selected)}/{len(cls_files)} to {target_dir}")
    return classes

for split in ["train", "valid", "test"]:
    download_subset(split, max_per_class=50)

print("\nDataset ready!")
for split in ["train", "valid", "test"]:
    split_dir = os.path.join(DATA_DIR, split)
    if os.path.exists(split_dir):
        total = 0
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                n = len([f for f in os.listdir(cls_dir) if f.endswith(".jpg")])
                print(f"  {split}/{cls}: {n} images")
                total += n
        print(f"  {split} total: {total}")
