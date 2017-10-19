import numpy as np

class KMeansClustering:

    def dist(self, x1, x2):
        """Eucledian Distance
            x1  and x2 are vector points
        """
        diff = x1 - x2
        diff_sq = diff ** 2
        sum_diff = diff_sq.sum()
        return np.sqrt(sum_diff)

    def fit(self, x_train, y_train = None, nIter = 100, k = 2):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self._labels = {}
        self.cluster_centers = {}
        if y_train == None:
            y_train = [x for x in range(k)]
        for i in range(k):
            self.cluster_centers[y_train[i]] = {
                'center' : np.random.random(x_train.shape[1]), # np.amax(x_train, axis = 0),
                'pts' : []
            }
        for i in range(nIter):
            for ix in range(x_train.shape[0]):
                distances = []
                for cx in self.cluster_centers.keys():
                    comp_dist = self.dist(x_train[ix, :], self.cluster_centers[cx]['center'])
                    distances.append([comp_dist, cx])
                best_dist = sorted(distances)[0]
                best_center = best_dist[1]
                self.cluster_centers[best_center]['pts'].append(x_train[ix])
                self._labels[ix] = best_center
            for cx in self.cluster_centers.keys():
                if not len(self.cluster_centers[cx]['pts']) == 0:
                    points = np.asarray(self.cluster_centers[cx]['pts'])
                    self.cluster_centers[cx]['center'] = points.mean(axis = 0)
                if i < nIter - 2:
                    self.cluster_centers[cx]['pts'] = []
        self._centers = np.array([self.cluster_centers[cx]['center'] for cx in self.cluster_centers.keys()])

    def predict(self, x_test):
        distances = []
        for cx in self.cluster_centers.keys():
            comp_dist = self.dist(self.cluster_centers[cx]['center'], x_test)
            distances.append([comp_dist, cx])
        best_dist = sorted(distances)[0]
        best_center = best_dist[1]
        return best_center


if __name__ == '__main__':
    mean_01 = np.array([2.0, 4.0])
    mean_02 = np.array([-1.0, 0.0])

    cov_01 = np.array([[1.0, 0.0], [0.0, 1.0]])
    cov_02 = np.array([[0.9, -0.4], [-0.4, 0.9]])

    dt1 = np.random.multivariate_normal(mean_01, cov_01, 250)
    dt2 = np.random.multivariate_normal(mean_02, cov_02, 250)
    
    print(dt1[:10, :])
    print(dt2[:10, :])
    
    all_data = np.zeros((500, 3))
    all_data[:250, :2] = dt1
    all_data[250:, :2] = dt2
    all_data[250:, -1] = 1
    
    print(all_data[:10, :])
    kMeans = KMeansClustering()
    
    split = int(0.8 * all_data.shape[0])
    np.random.shuffle(all_data)
    
    from matplotlib import pyplot as plt
    
    plt.figure(0)
    plt.scatter(all_data[:, :1], all_data[:, 1:2])
    plt.show()
    
    kMeans.fit(all_data[:, :2])
    cols = ['red', 'green']
    plt.figure(0)
    for cx in kMeans.cluster_centers.keys():
        if not len(kMeans.cluster_centers[cx]['pts'])==0:
            points = np.asarray(kMeans.cluster_centers[cx]['pts'])
            print(points.shape)
            plt.scatter(points[:, 0], points[:, 1], color=cols[cx])
    plt.show()