import numpy as np


def entropy(row):
    labels, counts = np.unique(row, return_counts=True)
    pis = counts / np.sum(counts)
    return -np.sum(pis * np.log2(pis))


if __name__ == '__main__':
    is_male = np.zeros(14)
    is_male[:9] = 1
    np.random.shuffle(is_male)
    assert round(entropy(is_male), 4) == 0.9403