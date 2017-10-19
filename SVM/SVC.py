import numpy as np
import matplotlib.pyplot as plt


class SVC(object):

    def __init__(self, n_epochs=100, kernel='linear', alpha=0.1):
        self.n_epochs = n_epochs
        self.kernel = kernel
        self.alpha = alpha

    def calc_error(self, x, y):
        if (y * np.dot(x, self.w)) < 1:
            return 1
        return 0

    def fit(self, X, Y, labels):
        self.labels = labels
        self.w = np.zeros(X[0].shape[0])

        self.errors = []

        for epoch in range(1, self.n_epochs+1):
            error = 0
            for i, x in enumerate(X):
                error = self.calc_error(x, Y[i])
                regularizer = -2 * (1 / epoch) * self.w
                self.w += self.alpha * (error * (x * Y[i]) + regularizer)
            self.errors.append(error)

        plt.plot(self.errors, '|')
        plt.ylim(0.5, 1.5)
        plt.axes().set_yticklabels([])
        plt.xlabel('Epoch')
        plt.ylabel('Missclassified')
        plt.show()

    def predict(self, X):
        preds = []
        for i, x in enumerate(X):
            if np.dot(x, self.w) >= 1:
                preds.append(self.labels[1])
            else:
                preds.append(self.labels[0])
        return preds

if __name__ == '__main__':
    X = np.array([
        [-2,4,-1],
        [4,1,-1],
        [1, 6, -1],
        [2, 4, -1],
        [6, 2, -1],
    ])

    Y = np.array([-1,-1,1,1,1])
    svc = SVC(n_epochs=10000, alpha=1)
    svc.fit(X, Y, [-1, 1])
    w = svc.w

    plt.figure(1)
    for d, sample in enumerate(X):
        # Plot the negative samples
        if d < 2:
            plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
        # Plot the positive samples
        else:
            plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

    # Add our test samples
    plt.scatter(2,2, s=120, marker='_', linewidths=2, color='yellow')
    plt.scatter(4,3, s=120, marker='+', linewidths=2, color='blue')

    # Print the hyperplane calculated by svm_sgd()
    x2=[w[0],w[1],-w[1],w[0]]
    x3=[w[0],w[1],w[1],-w[0]]

    x2x3 =np.array([x2,x3])
    X,Y,U,V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X,Y,U,V,scale=1, color='blue')
    plt.show()

    print(svc.predict(np.array([[2, 2, -1], [4, 3, -1]])))