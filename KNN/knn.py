import numpy as np

class KNN:
    def __init__(self, k=5, dist='eucledian'):
        self.k = k
    
    def sim_eucledian(self, x1, x2):
        return np.sqrt(((x1-x2)**2).sum(axis=1))
    
    def fit(self, x_train, y_train):
        self.xtr = x_train
        self.ytr = y_train
    
    def predict(self, x_test):
        preds = []
        for ix in range(x_test.shape[0]):
            dist = self.sim_eucledian(self.xtr, x_test[ix])
            dist = np.array([[dist[i], self.ytr[i]] for i in range(dist.shape[0])])
            k_neighbours = np.array(sorted(dist, key=lambda x:x[0])[:self.k])
            labels = k_neighbours[:, -1]
            freq = np.unique(labels, return_counts=True)
            preds.append(freq[0][freq[1].argmax()])
        return np.array(preds)
    
    def accuracy(self, x_test, y_true):
        preds = self.predict(x_test)
        y_true = np.array(y_true)
        accuracy = ((preds == y_true).sum()) / y_true.shape[0]
        return accuracy


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
    
    
    kn = None
    accuracy = []
    k_test = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for ik in k_test:
        kn = KNN(k = ik)
        kn.fit(x_train, y_train)
        kn.predict(x_test)
        acc = kn.accuracy(x_test, y_test)
        accuracy.append(acc)
        print('k:', ik, '\tAccuracy:', acc)
    
    plt.figure(0)
    plt.plot(k_test, accuracy, 'r-*')
    plt.show()