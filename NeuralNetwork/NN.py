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


class Model(object):
    LOSSES_DICT = {
        'Sigmoid_Loss': sigmoid_logistic_loss
    }

    def __init__(self, input_shape, n_epoch=100,
                 learning_rate=0.01, loss_type='Sigmoid_Loss'):
        # input stacked column wise a colum represents an example
        # shape of input - num features X num examples
        """
        Initialize a Model object that stacks layes one over other.
        @params:
            input_shape: shape of input - format followed (num-features, num-examples)
            n_epoch: Number of epochs
            learning_rate: Rate at which we descent the slope at a given epoch
            loss_type: The loss function to be used
        @attrbiutes:
            layers: list of layer objects
            X: input (features) for training
            Y: output (labels) for training
            AL: Output of the last layer after a forward pass
        """
        self.input_shape = input_shape
        self.layers = []
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.loss_type = loss_type

    def add_layer(self, outputs, activation='sigmoid'):
        """
        Add a layer object on the model's stack.
        @params:
            outputs: Number of units or number of outputs
            activation: Name of activation to be used.
        """
        if len(self.layers) == 0:
            layer = Layer(self.input_shape[0], outputs,
                          activation=activation)
        else:
            layer = Layer(self.layers[-1].output_size, outputs,
                          activation=activation)

        self.layers.append(layer)

    def _calc_loss(self):
        return np.squeeze(Model.LOSSES_DICT[self.loss_type](self.AL, self.Y))

    def _forward_pass_(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward_op(out)
        self.AL = out
        return out

    def _backward_pass_(self):
        dAL = Model.LOSSES_DICT[self.loss_type](self.AL, self.Y, derive=True)
        dA_prev = dAL
        for l in reversed(range(len(self.layers))):
            dA_prev = self.layers[l].backward_op(dA_prev)
            self.layers[l].update_params(self.learning_rate)

    def train(self, X, Y, print_cost=False, print_at_epoch=None):
        """
        Train the model on given data
        @params:
            X: train features
            Y: train labels
            print_cost: if True print_cost (at every print_at_epoch)
            print_at_epoch: At every print_at_epoch the cost will be printed
        """
        if print_cost:
            assert print_at_epoch != None
        self.X = X
        self.Y = Y
        for i in range(self.n_epoch):
            self._forward_pass_(X)
            self._backward_pass_()
            if print_cost and i % print_at_epoch == 0:
                print('Epoch: {}, Loss: {}'.format(i, self._calc_loss()))

    def predict(self, X, prob_pred=False, threshold=0.5):
        """
        Make predictions on given input X.
        @params:
            X: to make predictions on 
            prob_pred: if True return the activation values else return pred > threshold array
        """
        if prob_pred:
            return self._forward_pass_(X)
        return self._forward_pass_(X) > threshold

    def accuracy(self, x_test, y_test):
        """
        Get the accuracy of the network on given data:
        @params:
            x_test: The test input features
            y_test: the test output features
        """
        return np.sum(self.predict(x_test) == y_test) / x_test.shape[1]

    def summary(self):
        """
        Print the neural network's layer shape and the type of network.
        """
        print('Input Shape: ', self.input_shape)
        print('-'*70)
        for il in range(len(self.layers)):
            print('Dense Layer', il)
            print(self.layers[il].weights.shape)
            print('-'*70)
        print('\nModel Type: ' + ('Binary Classifier' 
                                  if self.layers[-1].output_size==1 else
                                  'Multi-Class Classifier'))
        print('Loss Metric used: ' + self.loss_type, end='\n\n')


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    
    from utils import plot_decision_boundary, load_dataset

    X, Y = load_dataset(700, 2)
    split = int(0.8 * X.shape[1])
    x_train, y_train = X[:, :split], Y[:, :split]
    x_test, y_test = X[:, split:], Y[:, split:]
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    
    np.random.seed(3)
    a = Model(x_train.shape, n_epoch=10000, learning_rate=1.2)
    a.add_layer(5, 'tanh')
    a.add_layer(1)
    a.summary()

    a.train(x_train, y_train, print_cost=True, print_at_epoch=1000)
    print('Accuracy on dev set: {}%'.format(round(a.accuracy(x_test, y_test)*100, 2)))
    plot_decision_boundary(lambda x: a.predict(x.T), X, Y)
