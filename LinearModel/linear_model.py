import numpy as np

class LinearRegression:

    def __init__(self, fit_intercept = True, normalize = False, copy_X = True, n_jobs = 1):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
    
    def normalize_data(self):
        x_temp = self.X
        y_temp = self.y
        
        x_temp = x_temp ** 2
        y_temp = y_temp **2
        
        x_temp.sum(axis = 0)
        y_temp.sum(axis = 0)
        
        x_temp.sqrt()
        y_temp.sqrt()
        
        self.X = x_temp
        self.y = y_temp
    
    def calc_theta1(self):
        xi_mean = self.X - self.x_mean
        yi_mean = self.y - self.y_mean
        
        mul_sum = np.multiply(xi_mean, yi_mean).sum()
        result = mul_sum / (xi_mean ** 2).sum()
        return result
    
    def calc_theta0(self):
        return self.y_mean - self.theta1 * self.x_mean
    
    def fit(self, x_test, y_test):
        self.X = np.asarray(x_test)
        self.y = np.asarray(y_test)
        self.x_mean = self.X.mean()
        self.y_mean = self.y.mean()
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.theta = np.zeros((1, self.X.shape[1]))
        
        if self.normalize:
            normalize_data()
        
        self.theta1 = self.calc_theta1()
        if self.fit_intercept:
            self.theta0 = self.calc_theta0()
    
    def predict(self, x_test):
        delta1 = x_test * self.theta1
        self.preds = delta1 + self.theta0
        return self.preds
    
    def accuracy(self, y_test):
        y_test = np.asarray(y_test)
        diff = self.preds - y_test
        diff_sq = diff ** 2
        sum_diff = diff_sq.sum()
        rmse = np.sqrt(sum_diff / len(y_test))
        return 1 - rmse


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    
    dataframe = pd.read_fwf('brain_body.txt', sep = ' ')
    x_values = dataframe[['Brain']]
    y_values = dataframe[['Body']]
    
    body_reg = LinearRegression()
    body_reg.fit(x_values, y_values)
    preds = body_reg.predict(x_values)
    
    print('Accuracy:', body_reg.accuracy(y_values))
    
    plt.figure(0)
    plt.scatter(x_values, y_values)
    plt.plot(x_values, preds)
    plt.show()