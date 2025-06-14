import numpy as np

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # moment 1
        self.v = {}  # moment 2
        self.t = 0   # timestep

    def step(self, params, grads):
        self.t += 1

        for key in params:
            param_id = id(params[key])  # dùng id để phân biệt các layer có cùng key như "W"

            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(grads[key])
                self.v[param_id] = np.zeros_like(grads[key])

            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grads[key]
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grads[key] ** 2)

            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)

            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
