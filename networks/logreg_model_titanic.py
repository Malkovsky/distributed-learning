import numpy as np


class LogRegTitanic:
    def __init__(self, dim, lr=5e-4, tau=1e-4):
        self.W = np.zeros(dim)
        self.lr = lr
        self.tau = tau

    def parameters(self):
        return self.W

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, x_train, y_train):
        grad = -np.array([
            np.dot(y_train * self._sigmoid(-y_train * (x_train @ self.W)), x_train[:, j])
            for j in range(x_train.shape[1])
        ]) / x_train.shape[0] + self.tau * self.W
        self.W -= self.lr * grad

        return self.calc_loss(x_train, y_train)

    def calc_accuracy(self, x_test, y_test):
        test_predictions = (self._sigmoid(x_test @ self.W) >= 0.5).astype(np.int) * 2 - 1
        return np.mean(test_predictions == y_test)

    def calc_loss(self, x, y):
        loss = self.tau / 2 * np.sum(self.W ** 2) + -np.mean(
            np.log(self._sigmoid(y * (x @ self.W))))
        return loss
