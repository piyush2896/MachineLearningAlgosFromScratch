import numpy as np


class GuassianNB(object):

    def _pdf(self, x_test, class_label=None):
        if class_label != None:
            mean = self.class_summary[class_label]['mean']
            sd = self.class_summary[class_label]['sd']
        else:
            mean = self.summary['mean']
            sd = self.summary['sd']
        den = 1 / (np.sqrt(2 * np.pi) * sd)
        num = np.exp(-((x_test - mean)**2) / (2 * sd**2))
        return num * den

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

    def predict(self, x_test):
        preds = []
        for label in self.labels:
            preds.append(np.prod(self._pdf(x_test, label), axis=1))
        preds = np.array(preds)
        return self.labels[np.argmax(preds, axis=0)]

if __name__ == '__main__':
    pass
