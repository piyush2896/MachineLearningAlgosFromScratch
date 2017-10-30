import numpy as np
from activations import *
from losses import *


class Layer(object):
    ACT_DICT = {
        'sigmoid': sigmoid,
        'relu': relu,
        'tanh': tanh,
        'softmax': softmax
    }

    def __init__(self, input_size, output_size,
                 random_bias=False, activation='sigmoid'):
        """
        Initialize a Dense Layer. Containing bias and 
        randomly initialized weights
        @params:
            input_size: Number of features
            output_size: Number of outputs (No. of Neurons) - 
            weights attrbiute initializes to a random 2d array of shape: (output_size, input_size)
            random_bias: Boolean value defining bias attrbiute to be 
            either random (True) or all zeros(False) of shape: (output_size, 1)
            activation: string name of the activation
        @attributes:
            Z: linear forward pass Value
            a_prev: Input Value from previous layer
            A: Activation forward pass Value
            dW: weight gradients
            db: bias gradients
            + @params
        """
        self.input_size = input_size
        self.output_size = output_size
        self.bias = np.random.rand(output_size, 1) * 0.01 if random_bias else np.zeros((output_size, 1))
        self.weights = np.random.rand(output_size, input_size) * 0.01
        self.activation = activation

    def forward_op(self, a_prev, apply_act=True):
        """
        Forward operation comprise of:
            self.Z = W.X + b
            self.A = g(Z), where g(.) is a non-lineratiy
        @params:
            a_prev: output of previous layer (X) in case of first layer (cached to self.a_prev)
            apply_act: if True apply activation function to Z.
        @returns:
            apply_act: if True return A else return Z.
        """
        self.a_prev = a_prev
        self.Z = np.dot(self.weights, self.a_prev) + self.bias
        if apply_act:
            self.A = Layer.ACT_DICT[self.activation](self.Z)
            return self.A
        return self.Z

    def backward_op(self, dAl):
        """
        Backward operation comprise of:
            dZ = dAL * g(Z)
            self.dW = (dZ.(a_prev.T)) / m
            self.db = np.sum(dZ, axis=1, keepdims=True) / m (i.e. average over columns)
                m = number of train examples
            dA_prev = (W.T).dZ
        @params:
            dAl: propagated error of current layer
        @returns:
            dA_prev: error of previous layer
        """
        dZ = dAl * Layer.ACT_DICT[self.activation](self.Z, derive=True)
        m = self.a_prev.shape[1]
        self.dW = np.dot(dZ, self.a_prev.T) / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.weights.T, dZ)
        return dA_prev

    def update_params(self, learning_rate=0.01):
        """
        Update parameters - weights and bias of layer
            W := W - learning_rate * dW
            b := b - learning_rate * db
        @params:
            learning_rate: Rate at which we descent the slope at a given epoch
        """
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.db
