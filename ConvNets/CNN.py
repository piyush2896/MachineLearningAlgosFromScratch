import numpy as np


class Conv2d(object):

    def __init__(self, W, strides=(1, 1), pad='VALID'):
        self.W = W
        self.strides = strides
        self.pad = pad

    def _apply_padding(self, X):
        pass

    def _output_shape(self):
        pass

    def forward_pass(self, X):
        pass

    def backward_pass(self):
        pass


