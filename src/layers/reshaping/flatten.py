import numpy as np
from ..base import Layer

class Flatten(Layer):
    def forward(self, x: np.ndarray):
        """
        x: input (batch_size, d1, d2, ..., dn)
        return: (batch_size, d1 * d2 * ... * dn)
        """
        self.input_shape = x.shape  # lưu lại để dùng khi backward
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: np.ndarray):
        """
        grad_output: gradient từ layer sau, shape (batch_size, D)
        return: reshape lại về input shape ban đầu
        """
        return grad_output.reshape(self.input_shape)
