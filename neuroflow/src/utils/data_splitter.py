import os
import shutil
import random
from typing import Tuple

class DatasetSplitter:
    def __init__(self, src_dir: str, dest_dir: str, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        assert train_ratio + val_ratio + test_ratio == 1.0, "Tổng tỉ lệ phải bằng 1"
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self):
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)

        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.dest_dir, split), exist_ok=True)

        for class_name in os.listdir(self.src_dir):
            class_path = os.path.join(self.src_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            images = os.listdir(class_path)
            random.shuffle(images)

            n_total = len(images)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)

            split_indices = {
                'train': images[:n_train],
                'val': images[n_train:n_train + n_val],
                'test': images[n_train + n_val:]
            }

            for split, image_list in split_indices.items():
                target_class_dir = os.path.join(self.dest_dir, split, class_name)
                os.makedirs(target_class_dir, exist_ok=True)
                for img in image_list:
                    src_img_path = os.path.join(class_path, img)
                    dst_img_path = os.path.join(target_class_dir, img)
                    shutil.copy2(src_img_path, dst_img_path)


# splitter = DatasetSplitter(
#     src_dir="./dataset",
#     dest_dir="./data"
# )
# splitter.split()
