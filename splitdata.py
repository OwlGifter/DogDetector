import os
import shutil
import random
from pathlib import Path

#splits dog and nodog images to train and validate folders
def split_dataset(base_dir='DogDetector/dataset/bulk data', split_ratio=0.8):
    categories = ['dog', 'no_dog']
    for category in categories:
        full_path = Path(base_dir) / category
        images = list(full_path.glob("*.jpg"))
        random.shuffle(images)

        train_count = int(len(images) * split_ratio)

        train_dir = Path(base_dir) / 'train' / category
        val_dir = Path(base_dir) / 'val' / category
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for img in images[:train_count]:
            shutil.move(str(img), train_dir / img.name)
        for img in images[train_count:]:
            shutil.move(str(img), val_dir / img.name)

    print("Dataset split complete.")

split_dataset()
