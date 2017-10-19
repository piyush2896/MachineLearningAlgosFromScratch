import numpy as np
import math
import operator


def entropy(rows):
    labels, counts = np.unique(rows, return_counts=True)
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


def divide_set(rows, attr, value, is_discrete=True):
    if is_discrete:
        left = rows[rows[:, attr] == value]
        right = rows[rows[:, attr] != value]
    else:
        left = rows[rows[:, attr] >= value]
        right = rows[rows[:, attr] < value]
    return left, right


if __name__ == '__main__':
    is_male = np.zeros(14)
    is_male[:9] = 1
    np.random.shuffle(is_male)
    assert round(entropy(is_male), 4) == 0.9403


    s = np.array([[1, 0], [1, 0], [0, 1], [1, 1]])
    assert round(information_gain(s, 0), 4) == 0.3113

    X = np.array([[1, 1, 1, 18], [2, 2, 1, 23],
                  [3, 1, 1, 24], [4, 2, 1, 23],
                  [2, 3, 2, 21], [5, 4, 2, 12],
                  [5, 3, 2, 21], [2, 1, 2, 24],
                  [1, 2, 1, 19], [3, 1, 2, 18],
                  [2, 3, 2, 18], [4, 3, 2, 19],
                  [3, 4, 1, 12], [1, 3, 2, 21],
                  [2, 3, 1, 18], [4, 2, 1, 19]])
    y = np.array([0, 2, 1, 1, 2, 0, 2, 2,
                  0, 0, 0, 0, 1, 0, 1, 1])

    test_set1 = np.array([[1, 1, 1, 18], [3, 1, 1, 24],
                          [2, 1, 2, 24], [3, 1, 2, 18]])
    test_set2 = np.array([[2, 2, 1, 23], [4, 2, 1, 23],
                          [2, 3, 2, 21], [5, 4, 2, 12],
                          [5, 3, 2, 21], [1, 2, 1, 19],
                          [2, 3, 2, 18], [4, 3, 2, 19],
                          [3, 4, 1, 12], [1, 3, 2, 21],
                          [2, 3, 1, 18], [4, 2, 1, 19]])

    set1, set2 = divide_set(X, 1, 1)
    assert set1.shape[0] * set1.shape[1] == test_set1.shape[0] * test_set1.shape[1]
    assert set2.shape[0] * set2.shape[1] == test_set2.shape[0] * test_set2.shape[1]
    assert np.sum(test_set1 == set1) == set1.shape[0] * set1.shape[1]
    assert np.sum(test_set2 == set2) == set2.shape[0] * set2.shape[1]
