import numpy as np
from typing import List
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod
from typing import Tuple, List


class NeuralNetwork:

    def __init__(self, num_neurons: List[int]):
        self.learning_rate = 0.1
        self.training_iters = 1
        self.layers = []
        for index, neuron in enumerate(num_neurons[:-1]):
            layer = LinearLayer((neuron, num_neurons[index+1]))
            self.layers.append(layer)

    def fit(self, x: np.ndarray, y: np.ndarray,
            learning_rate: float = 0.0001,
            training_iters: int = 1000) -> np.array:

        x = np.insert(x, 0, 1, 1)

        loss_layer = SqrErrorLayer(y)
        self.layers.append(loss_layer)

        losses = np.empty(self.training_iters)
        for i in range(self.training_iters):
            print(f"y: {y}")
            print(f"x: {x}")

            
            grad_inputs = self.layers[-1].backward()
            #for layer in
            z2 = self.linear_layer.forward(x)
            print(f"x+1: {z2}")
            loss = self.loss.forward(z2)
            print(f"loss: {loss}")
            d2 = self.loss.backward()
            print(f"loss backward: {d2}")
            d3 = self.linear_layer.backward(d2)
            print(f" linear backward: {d3}")
            gradient = self.linear_layer.grad_weights
            gradient *= self.learning_rate

            self.linear_layer.weights -= gradient

        return losses

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.insert(x, 0, 1, 1)
        for layer in self.layers:
            x = layer.forward(x)
        return x


class LinearLayer:

    def __init__(self, shape: Tuple[int, int]):
        self.weights = np.random.randn(shape[0], shape[1])

    def forward(self, x: np.ndarray):
        self.x = x
        self.next_x = np.dot(x, self.weights)
        return self.next_x

    def backward(self, next_grad_inputs: np.ndarray):
        self.grad_weights = np.dot(self.x.T,
                                   next_grad_inputs) / next_grad_inputs.shape[0]
        grad_inputs = next_grad_inputs.dot(self.weights.transpose())
        return grad_inputs


class SqrErrorLayer:

    def __init__(self, y: np.ndarray):
        self.y = y

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.next_x = np.mean(np.square(x-self.y), axis=0, keepdims=True)
        return self.next_x

    def backward(self) -> np.ndarray:
        grad_inputs = -1*(self.y - self.x)
        return grad_inputs


class SigmoidLayer:

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.next_x = 1/(np.exp(-x)+1)
        return self.next_x

    def backward(self, next_grad_inputs: np.ndarray) -> np.ndarray:
        return (self.next_x.T*(1-self.next_x).T)*next_grad_inputs
        

if __name__ == "__main__":
    np.random.seed(0)
    model = NeuralNetwork(training_iters=1, learning_rate=0.01)
    x = np.array([[-10]])
    y = 5 + 3*x + np.random.normal(0, 3)
    errors = model.fit(x, y)
    print(y)
    print(model.predict(x)[:10])
