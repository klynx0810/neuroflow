import os
import shutil
import random
from typing import Tuple

from .augmentation import (
    flip_image,
    rotate_image,
    scale_and_translate,
    shear_image,
    random_crop,
    adjust_brightness_contrast,
    adjust_saturation,
    add_gaussian_noise,
    add_salt_pepper_noise,
    blur_image,
    barrel_distortion,
    add_light_leak
)
import cv2 as cv

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

    def split_and_augment(self, aug_per_image=10):
        self.split()  # Gọi hàm chia dữ liệu trước

        train_dir = os.path.join(self.dest_dir, "train")
        for class_name in os.listdir(train_dir):
            class_path = os.path.join(train_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for fname in os.listdir(class_path):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(class_path, fname)
                img = cv.imread(img_path)
                if img is None:
                    print(f"Lỗi đọc ảnh: {img_path}")
                    continue

                base_name, ext = os.path.splitext(fname)
                for i in range(aug_per_image):
                    aug = img.copy()
                    aug = flip_image(aug, horizontal=random.choice([True, False]))
                    aug = rotate_image(aug, angle=random.uniform(-20, 20))
                    aug = scale_and_translate(aug, scale=1.1, tx=5, ty=5)
                    aug = shear_image(aug, shear_factor=0.2)
                    aug = random_crop(aug, crop_size=(img.shape[0], img.shape[1]))
                    aug = adjust_brightness_contrast(aug)
                    aug = adjust_saturation(aug, saturation_factor=random.uniform(0.8, 1.5))
                    aug = add_gaussian_noise(aug, std=20)
                    aug = add_salt_pepper_noise(aug, amount=0.01)
                    aug = blur_image(aug, ksize=3)

                    if random.random() < 0.3:  # 30% xác suất
                        strength = random.uniform(0.00005, 0.0002)
                        aug = barrel_distortion(aug, strength=strength)
                    
                    if random.random() < 0.2:  # 20% xác suất
                        aug = add_light_leak(aug, intensity=0.4)

                    save_name = f"{base_name}_aug{i}{ext}"
                    save_path = os.path.join(class_path, save_name)
                    cv.imwrite(save_path, aug)


# splitter = DatasetSplitter(
#     src_dir="./dataset",
#     dest_dir="./data"
# )
# splitter.split()
