import numpy as np


def sigmoid_logistic_loss(y_out, y_true, derive=False):
    if derive:
        y_true.reshape(y_out.shape)
        dLoss = -((y_true / y_out) - ((1 - y_true) / (1 - y_out)))
        return dLoss

    loss = -(np.dot(y_true, np.log(y_out).T) +
             np.dot((1-y_true), np.log(1-y_out).T))
    return loss / y_out.shape[1]

