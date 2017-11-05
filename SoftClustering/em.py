import numpy as np

class EMClust(object):

    def __init__(self, k=8, n_iter=100):
        self.k = k
        self.n_iter = 100
        self.clust_centers = {}

    def _calc_prob(self, X, clust_key):
        variance = np.var(X, axis=1)
        clust_mean = self.clust_centers[clust_key]['mean']
        sq_dist = np.sum((X - clust_mean) ** 2, axis=1)
        power = -0.5 * variance * sq_dist
        return np.array([np.exp(power)]).T

    def _calc_new_mean(self, X):
        for ix in self.clust_centers:
            exps = self.clust_centers[ix]['expectations']
            self.clust_centers[ix]['mean'] = \
                        np.sum(X * exps, axis=0) / np.sum(exps, axis=0)

    def _init_clusts(self, X):
        clust_indices = np.random.randint(X.shape[0], size=self.k)
        for ix in range(clust_indices.shape[0]):
            self.clust_centers[ix] = {
                'mean': X[clust_indices[ix]],
            }

    def _total_exps(self, shape):
        total = np.zeros((shape[0],1))
        for ix in self.clust_centers:
            total += self.clust_centers[ix]['expectations']
        return total

    def fit(self, X):
        self._init_clusts(X)
        for iter in range(self.n_iter):
            for ix in self.clust_centers:
                probs = self._calc_prob(X, ix)
                self.clust_centers[ix]['expectations'] = probs
                #print(probs)
                #input()

            total_exps = self._total_exps(X.shape)
            for ix in self.clust_centers:
                self.clust_centers[ix]['expectations'] /= total_exps

            self._calc_new_mean(X)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    mean_01 = np.array([2.5, 1.0])
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

    clust = EMClust(k=2)
    clust.fit(all_data[:, :-1])
    X = all_data[:, :-1]
    plt.figure(0)
    count_in=0
    count_g = 0
    color = ['red', 'blue']
    for ix in clust.clust_centers:
        exps = clust.clust_centers[ix]['expectations']
        clust_center = clust.clust_centers[ix]['mean']
        pts = exps > 0.6
        count_in += np.sum(pts)
        pts_mid = (exps >=0.4) * (exps <=0.6)
        count_g += np.sum(pts_mid)

        plt.scatter(X[pts[:, 0], 0], X[pts[:, 0], 1], color=color[ix])
        plt.scatter(X[pts_mid[:, 0], 0], X[pts_mid[:, 0], 1], color='green')
        plt.scatter(clust_center[0], clust_center[1], color='black', marker='X')
    plt.show()
    print(count_in, count_g)
