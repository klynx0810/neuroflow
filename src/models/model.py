from ..layers.base import Layer
from typing import List
from ...registry import get_loss, get_optimizer
import numpy as np

class Model(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.layers: List[Layer] = []
        self.optimizer = None
        self.loss_fn = None

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

    def compile(self, optimizer, loss):
        """
        Thiết lập loss function và optimizer từ string hoặc callable
        """
        self.optimizer = get_optimizer(optimizer)
        self.loss_fn = get_loss(loss)

    def add(self, layer: Layer):
        assert isinstance(layer, Layer), f"{layer} không phải lớp Layer"
        self.layers.append(layer)

    def call(self, x: np.ndarray):
        """Thực hiện forward qua tất cả các layer"""
        for layer in self.layers:
            x = layer.forward(x)
            # print(f"{layer.name}: {x.shape}")
        return x

    def fit(self, X, y, epochs=1):
        for epoch in range(epochs):
            # 1. Forward
            y_pred = self.call(X)

            # 2. Loss
            loss = self.loss_fn(y_pred, y)

            # 3. Đạo hàm của loss theo y_pred
            if hasattr(self.loss_fn, "backward"):
                grad_output = self.loss_fn.backward(y_pred, y)
            else:
                raise NotImplementedError("Loss function phải có backward()")

            # 4. Truyền ngược
            for layer in reversed(self.layers):
                if hasattr(layer, "backward"):
                    grad_output = layer.backward(grad_output)

            # 5. Cập nhật trọng số qua optimizer
            for layer in self.layers:
                if hasattr(layer, "params") and hasattr(layer, "grads"):
                    self.optimizer.step(layer.params, layer.grads)

            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")


    def predict(self, X):
        return self.call(X)

    def evaluate(self, X, y):
        """
        Tính loss trên tập test/validation
        """
        preds = self.call(X)
        loss = self.loss_fn(preds, y)
        print(f"Evaluation Loss: {loss:.4f}")
        return loss