import numpy as np


def entropy(row):
    labels, counts = np.unique(row, return_counts=True)
    pis = counts / np.sum(counts)
    return -np.sum(pis * np.log2(pis))


def information_gain(S, A):
    labels, counts = np.unique(S[:, A], return_counts=True)
    avgs = counts / np.sum(counts)
    current_score = entropy(S[:, -1])
    weighted_avg = 0
    for i in range(labels.shape[0]):
        weighted_avg += avgs[i] * entropy(S[S[:, A]==labels[i]][:, -1])
    return current_score - weighted_avg


if __name__ == '__main__':
    is_male = np.zeros(14)
    is_male[:9] = 1
    np.random.shuffle(is_male)
    assert round(entropy(is_male), 4) == 0.9403


    s = np.array([[1, 0], [1, 0], [0, 1], [1, 1]])
    assert round(information_gain(s, 0), 4) == 0.3113