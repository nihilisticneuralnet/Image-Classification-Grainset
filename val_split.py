import os
import shutil
import random

train_dir = 'rice/train'
val_dir = 'rice/val1'

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

for subdir in os.listdir(train_dir):
    train_subdir_path = os.path.join(train_dir, subdir)
    val_subdir_path = os.path.join(val_dir, subdir)
    if not os.path.exists(val_subdir_path):
        os.makedirs(val_subdir_path)
    images = [f for f in os.listdir(train_subdir_path) if os.path.isfile(os.path.join(train_subdir_path, f))]
    num_val_images = int(0.1 * len(images))
    val_images = random.sample(images, num_val_images)
    for image in val_images:
        src = os.path.join(train_subdir_path, image)
        dest = os.path.join(val_subdir_path, image)
        shutil.move(src, dest)
    print(f"{num_val_images} images moved to {val_subdir_path}")

