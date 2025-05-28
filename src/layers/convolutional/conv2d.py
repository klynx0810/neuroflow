import numpy as np
from neuroflow.src.layers.base import Layer

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, stride=1, padding=0, input_shape=None, name=None):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding  # số pixel pad (int)
        self.input_shape = input_shape  # không cần thiết lắm nếu build theo x.shape

    def build(self, input_shape):
        self.batch_size, in_h, in_w, in_c = input_shape
        kh, kw = self.kernel_size
        # Khởi tạo W: (filters, kh, kw, in_c), mỗi filter dùng cho mọi kênh
        self.params["W"] = np.random.randn(self.filters, kh, kw, in_c) * 0.01
        self.params["b"] = np.zeros((self.filters,))
        self.built = True

    def conv2d_single_channel(self, A, W, bias, stride, pad):
        n_H_old, n_W_old = A.shape
        f, _ = W.shape
        A_pad = np.pad(A, pad_width=pad, mode='constant', constant_values=0)

        n_H_new = int((n_H_old - f + 2 * pad) / stride) + 1
        n_W_new = int((n_W_old - f + 2 * pad) / stride) + 1

        out = np.zeros((n_H_new, n_W_new))
        for i in range(n_H_new):
            for j in range(n_W_new):
                vert_start = i * stride
                horiz_start = j * stride
                out[i, j] = np.sum(A_pad[vert_start:vert_start+f, horiz_start:horiz_start+f] * W) + bias
        return out

    def forward(self, x):
        if not self.built:
            self.build(x.shape)

        self.last_input = x
        B, H, W, C = x.shape
        F, kh, kw, _ = self.params["W"].shape
        stride = self.stride
        pad = self.padding

        # Tính output shape
        out_h = int((H - kh + 2 * pad) / stride) + 1
        out_w = int((W - kw + 2 * pad) / stride) + 1
        out = np.zeros((B, out_h, out_w, F))

        for b in range(B):
            for f in range(F):
                out_channel = np.zeros((out_h, out_w))
                for c in range(C):
                    A = x[b, :, :, c]
                    Wf = self.params["W"][f, :, :, c]
                    out_channel += self.conv2d_single_channel(A, Wf, 0, stride, pad)
                out_channel += self.params["b"][f]
                out[b, :, :, f] = out_channel
        return out

    def backward(self, grad_output):
        raise NotImplementedError("Chưa triển khai Conv2D.backward")
