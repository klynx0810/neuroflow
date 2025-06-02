import numpy as np

class BinaryCrossentropy:
    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)
    
    def forward(self, y_true, y_pred):
        """
        y_true: (batch_size,) or (batch_size, 1)
        y_pred: (batch_size,) or (batch_size, 1) - đã sigmoid
        """
        self.y_true = y_true
        self.y_pred = y_pred
        eps = 1e-12
        loss = - (y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
        return np.mean(loss)

    def backward(self):
        """
        Gradient của loss theo y_pred
        """
        eps = 1e-12
        grad = (-(self.y_true / (self.y_pred + eps)) + ((1 - self.y_true) / (1 - self.y_pred + eps)))
        return grad / self.y_true.shape[0]
