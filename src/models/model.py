from ..layers.base import Layer
from typing import List

class Model(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.layers: List[Layer] = []
        self.optimizer = None
        self.loss_fn = None

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_fn = loss

    def add(self, layer: Layer):
        assert isinstance(layer, Layer), f"{layer} không phải lớp Layer"
        self.layers.append(layer)

    def call(self, x):
        """Thực hiện forward qua tất cả các layer"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def fit(self, X, y, epochs=1):
        for epoch in range(epochs):
            outputs = self.call(X)
            loss = self.loss_fn(outputs, y)

            # Giả lập cập nhật trọng số (nâng cấp sau)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

    def predict(self, X):
        return self.call(X)

    def evaluate(self, X, y):
        preds = self.call(X)
        loss = self.loss_fn(preds, y)
        print(f"Evaluation Loss: {loss:.4f}")
        return loss