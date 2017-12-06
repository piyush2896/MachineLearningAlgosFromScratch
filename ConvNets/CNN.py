import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor


class Conv2d(object):

    def __init__(self, W, strides=(1, 1), pad='VALID'):
        self.W = W
        self.strides = strides
        self.pad = pad
        self.b = np.random.randn(1, 1, 1, self.W.shape[3])

    def _calc_pad(self, X):
        in_height = X.shape[1]
        in_width = X.shape[2]
        filter_height = self.W.shape[0]
        filter_width = self.W.shape[1]

        pad_along_height = (in_height * (self.strides[0] - 1) -
                            self.strides[0] + filter_height) / 2
        self.pad_along_height = max(pad_along_height, 0)
        pad_along_width = (in_width * (self.strides[1] - 1) -
                           self.strides[1] + filter_width) / 2
        self.pad_along_width = max(pad_along_width, 0)

        return self.pad_along_height, self.pad_along_width

    def _apply_padding(self, X):
        (m, n_H_prev, n_W_prev, n_C) = X.shape
        pad_along_height, pad_along_width = self._calc_pad(X)
        print(pad_along_height, pad_along_width)
        n_H = n_H_prev + 2 * pad_along_height
        n_W = n_H_prev + 2 * pad_along_width
        self.X_pad = np.pad(X, ((0, 0), (ceil(pad_along_height), floor(pad_along_height)),
                                (ceil(pad_along_width), floor(pad_along_width)), (0, 0)),
                            'constant', constant_values=(0, 0))
        return self.X_pad

    def _output_shape(self):
        (m, in_height, in_width, n_C) = self.X.shape
        (f_height, f_width, n_C_prev, n_C) = self.W.shape
        out_height = int(1 + (in_height + 2 * self.pad_along_height -
                              f_height) / self.strides[0])
        out_width = int(1 + (in_width + 2 * self.pad_along_width -
                             f_width) / self.strides[1])
        return (m, out_height, out_width, n_C)

    def _convolve(self, slice, filter_num):
        return np.sum(slice * self.W[:, :, :, filter_num]) + self.b[:, :, :, filter_num]

    def forward_pass(self, X):
        self.X = X
        if self.pad == 'VALID':
            self.X_pad = self.X
        elif self.pad == 'SAME':
            self.X_pad = self._apply_padding(self.X)

        (m, in_height, in_width, n_C) = self.X.shape
        (f_height, f_width, n_C_prev, n_C) = self.W.shape

        out_shape = self._output_shape()
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


if __name__ == '__main__':
    conv = Conv2d(np.random.randn(3, 3, 2, 10))
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = conv._apply_padding(x)
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