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

    from matplotlib import pyplot as plt
    
    mean_01 = np.array([1.0, 2.0])
    mean_02 = np.array([-1.0, 4.0])
    
    cov_01 = np.array([[1.0, 0.9], [0.9, 2.0]])
    cov_02 = np.array([[2.0, 0.5], [0.5, 1.0]])
    
    ds_01 = np.random.multivariate_normal(mean_01, cov_01, 250)
    ds_02 = np.random.multivariate_normal(mean_02, cov_02, 250)
    
    all_data = np.zeros((500, 3))
    all_data[:250, :2] = ds_01
    all_data[250:, :2] = ds_02
    all_data[250:, -1] = 1
    
    np.random.shuffle(all_data)
    
    split = int(0.8 * all_data.shape[0])
    x_train = all_data[:split, :2]
    x_test = all_data[split:, :2]
    y_train = all_data[:split, -1]
    y_test = all_data[split:, -1]
    plt.figure(0)
    plt.scatter(x_train[y_train==0, 0], x_train[y_train==0, 1], color='red')
    plt.scatter(x_train[y_train==1, 0], x_train[y_train==1, 1], color='blue')
    plt.show()
    
    clf = GuassianNB()
    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_test)
    print('Accuracy: {}%'.format(np.sum(y_hat == y_test)/y_test.shape[0] * 100))
    
    x_min, x_max = all_data[:, 0].min() - 1, all_data[:, 0].max() + 1
    y_min, y_max = all_data[:, 1].min() - 1, all_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(0)
    plt.scatter(x_train[y_train==0, 0], x_train[y_train==0, 1], color='red')
    plt.scatter(x_test[y_hat==0, 0], x_test[y_hat==0, 1], color='red', marker='*')
    plt.scatter(x_train[y_train==1, 0], x_train[y_train==1, 1], color='blue')
    plt.scatter(x_test[y_hat==1, 0], x_test[y_hat==1, 1], color='blue', marker='+')
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
    plt.show()

