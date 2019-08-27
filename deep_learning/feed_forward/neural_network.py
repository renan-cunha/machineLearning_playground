import numpy as np
from typing import List
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod


class NeuralNetwork:

    def __init__(self, learning_rate: float = 0.0001, 
                 training_iters: int = 100):
        self.learning_rate = learning_rate
        self.training_iters = training_iters

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.array:
        x = np.insert(x, 0, 1, 1)

        self.weights = np.random.normal(0, 1, size=(2, 1))

        self.linear_layer = LinearLayer(self.weights)
        self.loss = ErrorLayer(y)

        losses = np.empty(self.training_iters)
        for i in range(self.training_iters):
            print(f"y: {y}")
            print(f"weights : {self.weights}")
            print(f"x: {x}")
            z2 = self.linear_layer.forward(x)
            print(f"x+1: {z2}")
            loss = self.loss.forward(z2)
            print(f"loss: {loss}")
            d2 = self.loss.backward()
            print(f"loss backward: {d2}")
            d3 = self.linear_layer.backward(d2)
            print(f" linear backward: {d3}")
            gradient = self.linear_layer.derivative
            gradient *= self.learning_rate

            self.weights -= gradient

        return losses

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.insert(x, 0, 1, 1)
        return np.dot(x, self.weights)


class LinearLayer:

    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def forward(self, x: np.ndarray):
        self.x = x
        self.next_x = np.dot(x, self.weights)
        return self.next_x

    def backward(self, gradient: np.ndarray):
        self.derivative = self.x.transpose().dot(gradient) / gradient.shape[0]
        self.out_derivative = gradient.dot(self.weights.transpose())
        return self.out_derivative


class ErrorLayer:

    def __init__(self, y: np.ndarray):
        self.y = y

    def forward(self, x: np.ndarray):
        self.x = x
        self.next_x = np.mean(np.square(x-self.y), axis=0, keepdims=True)
        return self.next_x

    def backward(self):
        self.out_derivative = -1*(self.y - self.x)
        return self.out_derivative


if __name__ == "__main__":
    np.random.seed(0)
    model = NeuralNetwork(training_iters=1, learning_rate=0.01)
    x = np.array([[-10]])
    y = 5 + 3*x + np.random.normal(0, 3)
    errors = model.fit(x, y)
    print(y)
    print(model.predict(x)[:10])
