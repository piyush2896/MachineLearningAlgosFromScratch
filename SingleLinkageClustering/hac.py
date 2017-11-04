import numpy as np

class Cluster(object):

    def __init__(self, initial_pts=None):
        if isinstance(initial_pts, (int, float)):
            self.points = np.array([[initial_pts]])
        else:
            self.points = initial_pts

    def add_points(self, points):
        if isinstance(self.points, np.ndarray):
            new_points = np.zeros((self.points.shape[0]+
                                  points.shape[0], self.points.shape[1]))
            new_points[:self.points.shape[0], :] = self.points
            new_points[self.points.shape[0]:self.points.shape[0]+points.shape[0], :] = points
        else:
            new_points = points
        self.points = new_points

    @staticmethod
    def merge_clusters(cluster1, cluster2):
        new_clust = Cluster(cluster1.points)
        new_clust.add_points(cluster2.points)
        return new_clust

    @staticmethod
    def eucledian_dist(points, point):
        return np.sqrt(np.sum((points - point) ** 2,
                              axis=1, keepdims=True))
    
    @staticmethod
    def min_dist_bw_clusters(cluster1, cluster2):
        min_dist_li = []
        for point in cluster2.points:
            dist_cur_pt = Cluster.eucledian_dist(cluster1.points,
                                                 point)
            min_dist_li.append(np.min(dist_cur_pt))
        return np.min(min_dist_li)

class SLC(object):

    def __init__(self, k=8):
        self.k = k
        self.clusters = None

    def fit(self, X_train):
        self.clusters = [Cluster(np.array([x])) for x in X_train]
        for iter in range(X_train.shape[0] - self.k):
            memo = {}
            for i in range(len(self.clusters)-1):
                for j in range(i+1, len(self.clusters)):
                    if not (((str(i) + ',' + str(j) in memo) or
                             (str(j) + ',' + str(i) in memo))):
                        memo[str(i) + ',' + str(j)] = \
                                Cluster.min_dist_bw_clusters(self.clusters[i],
                                                            self.clusters[j])
            memo_vals = np.array(list(memo.values()))
            memo_keys = list(memo.keys())
            closest_clusters = memo_keys[np.argmin(memo_vals)]
            pos_s = closest_clusters.split(',')
            new_clust = Cluster.merge_clusters(self.clusters[int(pos_s[0])],
                                               self.clusters[int(pos_s[1])])

            del self.clusters[int(pos_s[1])]
            del self.clusters[int(pos_s[0])]
            self.clusters.append(new_clust)
            print(iter, 'completed')
        return self.clusters


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    mean_01 = np.array([3.0, -2.0])
    mean_02 = np.array([-1.0, 4.0])
    
    cov_01 = np.array([[1.0, 0.9], [0.9, 2.0]])
    cov_02 = np.array([[2.0, 0.5], [0.5, 1.0]])
    
    ds_01 = np.random.multivariate_normal(mean_01, cov_01, 50)
    ds_02 = np.random.multivariate_normal(mean_02, cov_02, 50)
    
    all_data = np.zeros((100, 3))
    all_data[:50, :2] = ds_01
    all_data[50:, :2] = ds_02
    all_data[50:, -1] = 1
    
    np.random.shuffle(all_data)
    
    split = int(0.8 * all_data.shape[0])
    x_train = all_data[:split, :2]
    x_test = all_data[split:, :2]
    y_train = all_data[:split, -1]
    y_test = all_data[split:, -1]
    

    slc = SLC(2)
    clusts= slc.fit(x_train)
    clust1 = clusts[0]
    clust2 = clusts[1]
    plt.figure(0)
    plt.subplot(121)
    plt.scatter(x_train[y_train==0, 0], x_train[y_train==0, 1], color='red')
    plt.scatter(x_train[y_train==1, 0], x_train[y_train==1, 1], color='blue')
    plt.subplot(122)
    plt.scatter(clust1.points[:, 0], clust1.points[:, 1], color='red')
    plt.scatter(clust2.points[:, 0], clust2.points[:, 1], color='blue')
    plt.show()
