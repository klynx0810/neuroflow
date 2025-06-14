import os
import numpy as np
# import cupy as np
from PIL import Image
from typing import Tuple, List, Dict

class ImageDatasetLoader:
    def __init__(self, root_dir: str, image_size: Tuple[int, int] = (64, 64)):
        self.root_dir = root_dir
        self.image_size = image_size
        self.class_to_idx = self._build_class_index()

    def _build_class_index(self) -> Dict[str, int]:
        classes = sorted(entry for entry in os.listdir(self.root_dir)
                         if os.path.isdir(os.path.join(self.root_dir, entry)))
        return {cls_name: idx for idx, cls_name in enumerate(classes)}

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(self.root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_dir, fname)
                    image = Image.open(img_path).convert("RGB")
                    image = image.resize(self.image_size)
                    X.append(np.asarray(image, dtype=np.float32) / 255.0)  # chuẩn hóa về [0, 1]
                    y.append(label)
        return np.array(X), np.array(y)
