"""Multiple linear regressor with gradient descent"""
import numpy as np
from sklearn.metrics import mean_squared_error


class LinearRegressor:

    def __init__(self) -> None:
        self.thetas: np.ndarray
        self.bias: np.ndarray

    def fit(self, x: np.ndarray, y: np.ndarray,
            learning_rate: float = 0.00001, num_iters: int = 10000) -> np.ndarray:
        """We want to minimeze the MSE cost function"""
        length = x.shape[0]
        y = y.reshape(1, length)
        ones_array = np.ones(length).reshape(100, 1)
        x = np.concatenate((x, ones_array), axis=1)

        dimensions = x.shape[1]
        self.thetas = np.random.rand(1, dimensions)
        errors = np.empty(num_iters)

        for i in range(num_iters):
            predicted = self.predict(x, append_one=False)
            self.thetas = self.thetas - (learning_rate/length) * np.sum((predicted - y)*x.transpose(),axis=1)
            errors[i] = mean_squared_error(y, predicted)
        return errors

    def predict(self, x: np.ndarray, append_one: bool = True) -> np.ndarray:
        if append_one:
            length = x.shape[0]
            ones_array = np.ones(length).reshape(100, 1)
            x = np.concatenate((x, ones_array), axis=1)
            print(x.shape)
        return np.matmul(self.thetas, x.transpose())

    def get_params(self) -> np.ndarray:
        return self.thetas


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = np.linspace(1, 100, 100).reshape(100, 1)
    y = np.linspace(1, 100, 100).reshape(100, 1)
    z = y*x
    data = np.concatenate((x, y), axis=1)
    model = LinearRegressor()
    errors = model.fit(data, z, num_iters=10, learning_rate=0.0001)
    plt.plot(errors)
    plt.show()
    result = model.predict(data).reshape(100)
    plt.plot(z, label="Truth")
    plt.plot(result, label="predicted")
    plt.legend()
    plt.show()
    print(model.get_params())
