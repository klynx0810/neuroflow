from ..layers.base import Layer
from typing import List
from ...registry import get_loss, get_optimizer

class Model(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.layers: List[Layer] = []
        self.optimizer = None
        self.loss_fn = None

    def compile(self, optimizer, loss):
        self.optimizer = get_optimizer(optimizer)
        self.loss_fn = get_loss(loss)

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
            #Forward
            y_pred = self.call(X)

            #Compute loss
            loss = self.loss_fn(y_pred, y)

            #Compute gradient w.r.t output (dL/dy_pred)
            grad_output = 2 * (y_pred - y) / y.shape[0]  # đạo hàm MSE

            #Backward pass qua từng layer (ngược lại)
            for layer in reversed(self.layers):
                if hasattr(layer, 'backward'):
                    grad_output = layer.backward(grad_output)

            #Update parameters
            for layer in self.layers:
                if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                    if self.optimizer:
                        self.optimizer.step(layer.params, layer.grads)

            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

    def predict(self, X):
        return self.call(X)

    def evaluate(self, X, y):
        preds = self.call(X)
        loss = self.loss_fn(preds, y)
        print(f"Evaluation Loss: {loss:.4f}")
        return loss