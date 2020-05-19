import numpy as np
from typing import Tuple


class LinearLayer:

    def __init__(self, shape: Tuple[int, int]):
        self.weights = np.random.randn(shape[0], shape[1])
        self.biases = np.random.randn(shape[1], 1)

    def forward(self, x: np.ndarray):
        self.x = x
        self.next_x = (self.x @ self.weights) + self.biases.T
        return self.next_x

    def backward(self, next_grad_inputs: np.ndarray):
        self.grad_weights = self.x.T @ next_grad_inputs
        self.grad_biases = np.sum(next_grad_inputs, axis=0, keepdims=True).T
        return next_grad_inputs @ self.weights.T


class SqrErrorLayer:

    def __init__(self, y: np.ndarray):
        self.y = y

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.next_x = np.square(x-self.y)
        return self.next_x

    def backward(self, next_grad_inputs: np.ndarray) -> np.ndarray:
        grad_inputs = -(self.y - self.x)*2*next_grad_inputs
        return grad_inputs


class SigmoidLayer:

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.next_x = 1/(np.exp(-x)+1)
        return self.next_x

    def backward(self, next_grad_inputs: np.ndarray) -> np.ndarray:
        return (self.next_x*(1-self.next_x))*next_grad_inputs


