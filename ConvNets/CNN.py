import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor


def _calc_pad(X, W_shape, strides):
    in_height = X.shape[1]
    in_width = X.shape[2]
    filter_height = W_shape[0]
    filter_width = W_shape[1]

    pad_along_height = (in_height * (strides[0] - 1) -
                        strides[0] + filter_height) / 2
    pad_along_height = max(pad_along_height, 0)
    pad_along_width = (in_width * (strides[1] - 1) -
                       strides[1] + filter_width) / 2
    pad_along_width = max(pad_along_width, 0)

    return pad_along_height, pad_along_width


def _apply_padding(X, W_shape, strides):
    (m, n_H_prev, n_W_prev, n_C) = X.shape
    pad_along_height, pad_along_width = _calc_pad(X, W_shape, strides)
    n_H = n_H_prev + 2 * pad_along_height
    n_W = n_H_prev + 2 * pad_along_width
    X_pad = np.pad(X, ((0, 0), (ceil(pad_along_height), floor(pad_along_height)),
                       (ceil(pad_along_width), floor(pad_along_width)), (0, 0)),
                   'constant', constant_values=(0, 0))
    return X_pad, pad_along_height, pad_along_width


def _output_shape(X, W_shape, pads=[2, 2], strides=[1, 1]):
    (m, in_height, in_width, n_C) = X.shape
    (f_height, f_width, n_C_prev, n_C) = W_shape
    out_height = int(1 + (in_height + 2 * pads[0] -
                          f_height) / strides[0])
    out_width = int(1 + (in_width + 2 * pads[1] -
                         f_width) / strides[1])
    return (m, out_height, out_width, n_C)


class Conv2d(object):

    def __init__(self, W, strides=(1, 1), pad='VALID'):
        self.W = W
        self.strides = strides
        self.pad = pad
        self.b = np.random.randn(1, 1, 1, self.W.shape[3])

    def _convolve(self, slice, filter_num):
        return np.sum(slice * self.W[:, :, :, filter_num]) + self.b[:, :, :, filter_num]

    def forward_pass(self, X):
        self.X = X
        if self.pad == 'VALID':
            self.X_pad = self.X
        elif self.pad == 'SAME':
            (self.X_pad, self.pad_along_height, self.pad_along_width) = \
                                    _apply_padding(self.X, self.W.shape, self.strides)

        (m, in_height, in_width, n_C) = self.X.shape
        (f_height, f_width, n_C_prev, n_C) = self.W.shape

        out_shape = _output_shape(X, W.shape, [self.pad_along_height, self.pad_along_width], self.strides)
        (m, out_height, out_width, n_C) = out_shape
        self.Z = np.zeros(out_shape)

        for i in range(m):
            x_pad = self.X_pad[i]
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(n_C):

                        v_start = h + h * (self.strides[0] - 1)
                        v_end = v_start + f_height
                        h_start = w + w * (self.strides[1] - 1)
                        h_end = h_start + f_width

                        slice_pad = x_pad[v_start:v_end, h_start:h_end, :]
                        self.Z[i, h, w, c] = self._convolve(slice_pad, c)
        return self.Z

    def backward_pass(self, a_prev):
        pass


class Pool(object):

    def __init__(self, mode='max', f=(2, 2), strides=(1, 1), pad='VALID'):
        self.strides = strides
        self.f = f
        self.pad = pad
        self.mode = mode

    def forward_pass(self, X):
        self.X = X
        if self.pad == 'VALID':
            self.X_pad = self.X
        elif self.pad == 'SAME':
            (self.X_pad, self.pad_along_height, self.pad_along_width) = \
                                    _apply_padding(self.X, self.f, self.strides)

        (m, in_height, in_width, n_C) = self.X.shape
        out_shape = _output_shape(X, W, [self.pad_along_height, self.pad_along_width], self.strides)
        (m, out_height, out_width, n_C) = out_shape
        self.Z = np.zeros(out_shape)

        for i in range(m):
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(n_C):
                        v_start = h + h * (self.strides[0] - 1)
                        v_end = v_start + self.f[0]
                        h_start = w + w * (self.strides[1] - 1)
                        h_end = h_start + self.f[1]

                        slice = self.X_pad[i, v_start:v_end, h_start:h_end, c]
                        if mode == 'max':
                            self.Z[i, h, w, c] = np.max(slice)
                        if mode == 'average':
                            self.Z[i, h, w, c] = np.mean(slice)
        return self.Z

    def backward_pass(self):
        pass


if __name__ == '__main__':
    w = np.random.randn(3, 3, 2, 10)
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad, pad_along_height, pad_along_width = _apply_padding(x, w.shape, strides=[1, 1])
    print ("x.shape =", x.shape)
    print ("x_pad.shape =", x_pad.shape)
    print ("x[1,1] =", x[1,1])
    print ("x_pad[1,1] =", x_pad[1,1])

    fig, axarr = plt.subplots(1, 2)
    axarr[0].set_title('x')
    axarr[0].imshow(x[0,:,:,0], cmap='gray')
    axarr[1].set_title('x_pad')
    axarr[1].imshow(x_pad[0,:,:,0], cmap='gray')
    plt.show()

    np.random.seed(1)
    A_prev = np.random.randn(10,4,4,3)
    W = np.random.randn(2,2,3,8)
    conv = Conv2d(W, strides=(2, 2), pad='SAME')
    Z = conv.forward_pass(A_prev)
    
    print("Z's mean =", np.mean(Z))
    print("Z[3,2,1] =", Z[3,2,1])