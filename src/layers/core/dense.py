import numpy as np
from ..base import Layer

class Dense(Layer):
    def __init__(self, units, input_dim=None, name=None):
        super().__init__(name=name)
        self.units = units
        self.input_dim = input_dim

    def build(self, input_shape):
        input_dim = self.input_dim or input_shape[-1]
        self.params["W"] = np.random.randn(input_dim, self.units) * 0.01
        self.params["b"] = np.zeros((self.units,))
        self.built = True

    def forward(self, x):
        if not self.built:
            self.build(x.shape)
        self.last_input = x
        W = self.params["W"]
        b = self.params["b"]
        return x @ W + b
    
    def backward(self, grad_output):
        """
        grad_output: gradient từ layer phía sau (shape: [batch_size, units])
        """
        W = self.params["W"]
        x = self.last_input

        # Gradient w.r.t weights and bias
        self.grads["W"] = x.T @ grad_output      # shape: (input_dim, units)
        self.grads["b"] = np.sum(grad_output, axis=0)  # shape: (units,)

        # Gradient w.r.t input (truyền cho layer trước đó)
        grad_input = grad_output @ W.T           # shape: (batch_size, input_dim)
        return grad_input
