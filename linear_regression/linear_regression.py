"""Multiple linear regressor with gradient descent"""
import numpy as np
from sklearn.metrics import mean_squared_error
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D


class LinearRegressor:

    def __init__(self) -> None:
        self.thetas: np.ndarray

    def fit(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 10**-10,
            num_iters: int = 10**20) -> np.ndarray:
        """Finds the theta parameters that minimizes the derivative of the 
        MSE cost function

        Return the errors"""
        length = x.shape[0]
        dimensions = x.shape[1]
        
        x = self._append_ones(x)

        # initialize weights
        self.thetas = np.random.rand(1, dimensions+1)
        errors = np.empty(num_iters)
        
        for i in range(num_iters):
            predicted = self.predict(x, append_one=False)
            difference = predicted - y
            summation = np.sum(np.matmul(difference, x), axis=1)
            self.thetas -= (learning_rate/length) * summation
            errors[i] = mean_squared_error(y.reshape(1, y.shape[0]), predicted)
        return errors

    def _append_ones(self, x: np.ndarray) -> np.ndarray:
        """Appends one on the data for biases"""
        length = x.shape[0]
        ones_array = np.ones((length, 1))
        return np.concatenate((x, ones_array), axis=1)

    def predict(self, x: np.ndarray, append_one: bool = True) -> np.ndarray:
        if append_one:
            x = self._append_ones(x)
        x = x.transpose()
        return np.matmul(self.thetas, x)

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return thetas and biases"""
        return self.thetas, self.bias


if __name__ == "__main__":
    dataset = load_boston()

    #  getting average number of rooms per dwelling and the property-yax per
    #  10.000
    x = dataset.data[:, [5, 9]]
    y = dataset.target
    # preprocessing
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y.reshape(y.shape[0], 1))
    y = y.flatten()

    model = LinearRegressor()
    errors = model.fit(x, y, num_iters=10000, learning_rate=0.001)
    
    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    axes = Axes3D(fig)
    axes.scatter(xs=x[:, 0], ys=x[:, 1], zs=y)
        
    length = 10000
    x = np.linspace(0, 1, num=length)
    data = np.empty((length, 2))
    data[:, 0] = x
    data[:, 1] = x
    result = model.predict(data) 

    axes.plot(xs=x, ys=x, zs=result.flatten())
    axes.set_xlabel("Number of Rooms")
    axes.set_ylabel("Tax Value")
    axes.set_zlabel("Price")
    plt.show()
    
    plt.plot(errors)
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()
