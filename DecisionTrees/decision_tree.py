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


class Node(object):

    def __init__(self, attr=-1, value=None, results=None,
                 tb=None, fb=None, is_discrete=True):
        self.attr = attr
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb
        self.is_discrete = is_discrete


class DecisionTreeClassifier(object):

    def __init__(self, min_samples_split=2):
        self.root = None
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        all_data = np.zeros((X.shape[0], X.shape[1] + 1))
        all_data[:, :-1] = X
        all_data[:, -1] = y
        self.root = DecisionTreeClassifier.build_tree(all_data)

    def build_tree(data):
        if data.shape[0] == 0:
            return Node()
        current_score = entropy(data[-1])

        best_gain = 0
        best_criteria = None
        best_sets = None

        column_counts = data.shape[1] - 1
        is_discrete = True
        for col in range(column_counts):
            column_values = np.unique(data[:, col])
            if len(column_values) < 250:
                is_discrete = True
            else:
                is_discrete = False
            for value in column_values:
                set1, set2 = divide_set(data, col, value, is_discrete=is_discrete)

                p = set1.shape[0] / data.shape[0]
                gain = current_score - p * entropy(set1[:, -1]) - (1-p) * entropy(set2[:, -1])
                if gain > best_gain and set1.shape[0] > 0 and set2.shape[0] > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)
        if best_gain > 0:
            true_branch = DecisionTreeClassifier.build_tree(best_sets[0])
            false_branch = DecisionTreeClassifier.build_tree(best_sets[1])
            return Node(attr=best_criteria[0], value=best_criteria[1],
                        tb=true_branch, fb=false_branch, is_discrete=is_discrete)
        else:
            x, y = np.unique(data[:, -1], return_counts=True)
            return Node(results= {x[ix]: y[ix] for ix in range(x.shape[0])})

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for ix in range(X.shape[0]):
            pred = DecisionTreeClassifier._classify_(X[ix], self.root)
            preds[ix] = max(pred.items(), key=operator.itemgetter(1))[0]
        return preds

    def _classify_(x, tree):
        if tree.results != None:
            return tree.results
        v = x[tree.attr]
        branch = None
        if not tree.is_discrete:
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return DecisionTreeClassifier._classify_(x, branch)


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

    dsc = DecisionTreeClassifier()
    dsc.fit(X, y)
    assert dsc.predict(np.array([[5, 1, 1, 5]])) == 1.0