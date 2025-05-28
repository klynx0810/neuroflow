class Layer:
    def __init__(self, name=None, trainable=True) -> None:
        self.name = name or self.__class__.__name__.lower()
        self.trainable = trainable
        self.built = False
        self.params = {}
        self.grads = {}

    def build(self, input_shape):
        self.build()

    def forward(self, x):
        """Tính toán đầu ra từ đầu vào"""
        raise NotImplementedError("Layer phải định nghĩa forward()")

    def backward(self, grad_output):
        """Lan truyền gradient ngược lại"""
        raise NotImplementedError("Layer phải định nghĩa backward()")

    def get_params(self):
        return self.params

    def get_grads(self):
        return self.grads