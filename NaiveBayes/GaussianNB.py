import numpy as np


class GuassianNB(object):

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.summary = {
            'mean': np.mean(self.X, axis=0),
            'sd': np.std(self.X, axis=0)
        }
        self.labels = np.unique(self.y)
        self.class_summary = {}
        for label in self.labels:
            self.class_summary[label] = {
                'x': self.X[self.y == label],
                'mean': np.mean(self.X[self.y == label], axis=0),
                'sd': np.std(self.X[self.y == label], axis=0)
            }

if __name__ == '__main__':
    pass