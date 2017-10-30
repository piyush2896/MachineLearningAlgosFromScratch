import numpy as np


def sigmoid(t, derive=False):
    res = 1 / (1 + np.exp(-t))
    if derive:
        return res * (1 - res)
    return res


def relu(t, derive=False):
    if derive:
        res = np.zeros(t.shape)
        res[t > 0] = 1
        return res
    return np.maximum(0, t)


def tanh(t, derive=False):
    res = np.tanh(t)
    if derive:
        return 1 - res ** 2
    return res


def softmax(t, derive=False):
    shift_t = t - np.max(t)
    exps = np.exp(shift_t)
    res = exps / np.sum(exps)
    if derive:
        pass
    return res

