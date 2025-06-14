from ..layers.base import Layer
from typing import List
from ...registry import get_loss, get_optimizer, get_metrics
import numpy as np
from tqdm import tqdm
from tqdm import trange
from ..saving import saving_api
import matplotlib.pyplot as plt

class Model(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.layers: List[Layer] = []
        self.optimizer = None
        self.loss_fn: Layer = None
        self.history = {}

    @property
    def _params(self):
        """
        Trả về tất cả params của các lớp con dưới dạng dict flatten
        Ví dụ: {dense.w: ..., dense.b: ..., conv2d.w: ...}
        """
        all_params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "params"):
                for k, v in layer.params.items():
                    all_params[f"{layer.name}.{k}"] = v
        return all_params

    def compile(self, optimizer, loss, metrics = None):
        """
        Thiết lập loss function và optimizer từ string hoặc callable
        """
        self.optimizer = get_optimizer(optimizer)
        self.loss_fn = get_loss(loss)
        self.metrics = get_metrics(metrics) if metrics else None

    def add(self, layer: Layer):
        assert isinstance(layer, Layer), f"{layer} không phải lớp Layer"
        self.layers.append(layer)

    def call(self, x: np.ndarray, training=True):
        """Thực hiện forward qua tất cả các layer"""
        for layer in self.layers:
            if "training" in layer.forward.__code__.co_varnames:
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
            # print(f"{layer.name}: {x.shape}")
        return x

    # def set_training(self, mode: bool):
    #     for layer in self.layers:
    #         if hasattr(layer, "set_training"):
    #             layer.set_training(mode)

    # def fit(self, X, y, epochs=1):
    #     for epoch in range(epochs):
    #         # 1. Forward
    #         y_pred = self.call(X)

    #         # 2. Loss
    #         loss = self.loss_fn(y_pred, y)

    #         # 3. Đạo hàm của loss theo y_pred
    #         if hasattr(self.loss_fn, "backward"):
    #             grad_output = self.loss_fn.backward(y_pred, y)
    #         else:
    #             raise NotImplementedError("Loss function phải có backward()")

    #         # 4. Truyền ngược
    #         for layer in reversed(self.layers):
    #             if hasattr(layer, "backward"):
    #                 grad_output = layer.backward(grad_output)

    #         # 5. Cập nhật trọng số qua optimizer
    #         for layer in self.layers:
    #             if hasattr(layer, "params") and hasattr(layer, "grads"):
    #                 self.optimizer.step(layer.params, layer.grads)

    #         print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

    def fit(self, X, y, epochs=1, batch_size=32):
        num_samples = X.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))

        # Khởi tạo log
        self.history = {"loss": []}
        if self.metrics:
            self.history[self.metrics.name] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_metric = 0.0
            print(f"\nEpoch {epoch+1}/{epochs}")
            with trange(num_batches, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as t:
                for i in t:
                    start = i * batch_size
                    end = min(start + batch_size, num_samples)
                    X_batch = X[start:end]
                    y_batch = y[start:end]

                    # Forward
                    y_pred = self.call(X_batch)

                    # Loss
                    loss = self.loss_fn(y_batch, y_pred)
                    epoch_loss += loss

                    if self.metrics:
                        metric_value = self.metrics(y_batch, y_pred)
                        metric = self.metrics.name
                        epoch_metric += metric_value
                        t.set_postfix(loss=loss, **{metric: metric_value})
                    else:
                        t.set_postfix(loss=loss)

                    # Backward
                    grad_output = self.loss_fn.backward(y_batch, y_pred)
                    for layer in reversed(self.layers):
                        if hasattr(layer, "backward"):
                            grad_output = layer.backward(grad_output)

                    # Update
                    for layer in self.layers:
                        if hasattr(layer, "params") and hasattr(layer, "grads"):
                            self.optimizer.step(layer.params, layer.grads)

            avg_loss = epoch_loss / num_batches
            self.history["loss"].append(avg_loss)

            if self.metrics:
                avg_metric = epoch_metric / num_batches
                self.history[self.metrics.name].append(avg_metric)
                print(f"Epoch {epoch+1} Completed - Avg Loss: {avg_loss:.4f} - Avg {metric}: {avg_metric:.4f}")
            else:
                print(f"Epoch {epoch+1} Completed - Avg Loss: {avg_loss:.4f}")

    def predict(self, X):
        return self.call(X, training=False)

    def evaluate(self, X, y):
        """
        Tính loss trên tập test/validation
        """
        preds = self.call(X)
        loss = self.loss_fn(preds, y)
        print(f"Evaluation Loss: {loss:.4f}")

        if self.metrics:
            metric_value = self.metrics(y, preds)
            print(f"Evaluation {self.metrics.name.title()}: {metric_value:.4f}")
            return loss, metric_value
        
        return loss
    
    def save(self, filepath: str):
        """Lưu model vào file .h5"""
        saving_api.save_model_to_h5(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Load model từ file .h5"""
        return saving_api.load_model_from_h5(filepath, model_class=cls)
    
    def plot_metrics(self, smooth=True, smooth_factor=0.8):
        def smooth_curve(values, factor=0.8):
            smoothed = []
            for v in values:
                if smoothed:
                    smoothed.append(smoothed[-1] * factor + v * (1 - factor))
                else:
                    smoothed.append(v)
            return smoothed

        if not self.history:
            print("⚠️ Chưa có lịch sử huấn luyện.")
            return

        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Gom các key thành loss / acc groups
        loss_keys = [k for k in self.history if "loss" in k.lower()]
        acc_keys = [k for k in self.history if "acc" in k.lower()]

        # Trục trái: loss
        for i, key in enumerate(loss_keys):
            y = smooth_curve(self.history[key], smooth_factor) if smooth else self.history[key]
            ax1.plot(y, label=key, linestyle='-', linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.tick_params(axis='y')
        ax1.grid(True)

        # Trục phải: accuracy
        ax2 = ax1.twinx() if acc_keys else None
        if ax2:
            for i, key in enumerate(acc_keys):
                y = smooth_curve(self.history[key], smooth_factor) if smooth else self.history[key]
                ax2.plot(y, label=key, linestyle='--', linewidth=2)
            ax2.set_ylabel("Accuracy")
            ax2.tick_params(axis='y')

        # Gộp legend cả 2 trục
        lines = ax1.get_lines() + (ax2.get_lines() if ax2 else [])
        labels = [line.get_label() for line in lines]
        fig.legend(lines, labels, loc='upper right')

        plt.title("Training Progress")
        plt.tight_layout()
        plt.show()
