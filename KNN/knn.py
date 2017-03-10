import numpy as np
from sklearn.metrics import accuracy_score

class KNN:
    
    def dist(self, x1, x2):
        """
        Eucledian Distance
        x1 and x2 are vector points
        """
        diff = x1 - x2
        diff_sq = diff ** 2
        sum_diff = diff_sq.sum()
        return np.sqrt(sum_diff)
    
    def fit(self, x_train, y_train, k = 2):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self._labels = []
    
    def predict(self, x_test):
        preds = np.zeros(x_test.shape[0])
        for ex in range(x_test.shape[0]):
            dist_labels = []
            for ix in range(self.x_train.shape[0]):
                point_dist = self.dist(self.x_train[ix], x_test[ex])
                dist_labels.append([point_dist, self.y_train[ix]])
            dist_labels = sorted(dist_labels)
            neighbours = np.asarray(dist_labels[:self.k])[:, -1]
            labels = np.unique(neighbours, return_counts = True)
            preds[ex] = labels[0][labels[1].argmax()]
        self._labels = preds
        return self._labels
    
    def accuracy(self, y_test):
        return accuracy_score(y_test, self._labels) * 100


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
    
    
    kn = KNN()
    accuracy = []
    k_test = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for ik in k_test:
        kn.fit(x_train, y_train, k = ik)
        kn.predict(x_test)
        acc = kn.accuracy(y_test)
        accuracy.append(acc)
        print('k:', ik, '\nAccuracy:', acc)
    
    plt.figure(0)
    plt.plot(k_test, accuracy, 'r-*')
    plt.show()