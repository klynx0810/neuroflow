import numpy as np

class CategoricalCrossentropy:
    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)
    
    def forward(self, y_true, y_pred):
        """
        y_true: (batch_size, num_classes) - one-hot
        y_pred: (batch_size, num_classes) - output đã qua softmax
        """
        self.y_true = y_true
        self.y_pred = y_pred
        # để tránh log(0)
        eps = 1e-12
        loss = -np.sum(y_true * np.log(y_pred + eps), axis=1)
        return np.mean(loss)

    def backward(self):
        """
        Gradient của loss theo y_pred
        """
        # batch size
        batch_size = self.y_true.shape[0]
        return -self.y_true / (self.y_pred + 1e-12) / batch_size
        