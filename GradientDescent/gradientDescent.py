import numpy as np

class LinearRegression:
    
    def __init__(self, alpha = 0.0001, fit_intercept = True, n_iter = 5, normalize = False):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter
        self.normalize = normalize
    
    def normalize_dataset(self):
        x_temp = self.X
        y_temp = self.y
        
        x_mean = x_temp.mean(axis = 0)
        y_mean = y_temp.mean(axis = 0)
        
        diff_x = x_temp - x_mean
        diff_y = y_temp - y_mean
        
        range_x = x_temp.max(axis = 0) - x_temp.min(axis = 0)
        range_y = y_temp.max(axis = 0) - y_temp.min(axis = 0)
        
        x_temp = diff_x / range_x
        y_temp = diff_y / range_y
        
        self.X = x_temp
        self.y = y_temp
    
    def calc_loss(self):
        h_theta = np.dot(self.theta, np.transpose(self.X))
        
        diff = h_theta - self.y
        diff_sq = diff ** 2
        sum_diff = diff_sq.sum()
        
        j_theta = sum_diff / (2 * self.y.shape[2])
        return j_theta
    
    def calc_delta(self):
        h_theta = np.dot(self.theta, np.transpose(self.X))
        
        diff = h_theta - self.y
        diff_x_mul = np.dot(diff, self.X)
        diff_sum = diff_x_mul.sum(axis = 0)
        
        return diff_sum / len(self.y)
        
    def fit(self, x_train, y_train):
        self.X = np.asarray(x_train)
        self.y = np.asarray(y_train)
        
        if self.fit_intercept:
            self.X = np.c_[np.ones(self.X.shape[0]), self.X]
        
        self.theta = np.random.random((1, self.X.shape[1]))
        
        if self.normalize:
            normalize_dataset()
        
        for i in range(self.n_iter):
            self.theta = self.theta - self.alpha * self.calc_delta()
    
    def predict(self, x_test):
        if self.fit_intercept:
            x_test = np.c_[np.ones(x_test.shape[0]), x_test]
        return np.transpose(np.dot(self.theta, np.transpose(x_test)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    mean_01 = np.array([5.0, 6.8])
    cov_01 = np.array([[1.0, 0.7], [0.7, 0.5]])
    
    ds = np.random.multivariate_normal(mean_01, cov_01, 250)

    x_values = ds[:, 0]
    y_values = ds[:, -1]
    
    gd = LinearRegression(alpha = 0.03, n_iter = 1200)
    gd.fit(x_values, y_values)
    
    plt.figure(0)
    plt.scatter(x_values, y_values)
    preds = gd.predict(x_values)
    plt.plot(x_values, preds)
    plt.show()