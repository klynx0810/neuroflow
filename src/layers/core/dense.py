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
        W = self.params["W"]
        b = self.params["b"]
        return x @ W + b