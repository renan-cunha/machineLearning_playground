import numpy as np
from typing import List


class NeuralNetwork:

    def __init__(self, num_neurons: List[np.ndarray]):
        self._weights: List[np.naddaray] = []

        for index, num in enumerate(num_neurons[:-1]):
            weights = np.random.rand(num_neurons[index+1], num)
            biases = np.random.rand(num_neurons[index+1], 1)
            weights = np.concatenate((weights, biases), axis=1)
            self._weights.append(weights)

    def activation_function(self, x: np.ndarray) -> np.ndarray:
        return 1/(1 + np.exp(-x))

    def _append_ones(self, x: np.ndarray) -> np.ndarray:
        ones = np.ones((1, x.shape[1]))
        return np.concatenate((x, ones), axis=0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:


    def predict(self, x: np.ndarray) -> np.ndarray:
        x = x.transpose()
        for weight in self._weights:
            x = self._append_ones(x)
            x = np.matmul(weight, x)
            x = self.activation_function(x)
        return x


if __name__ == "__main__":
    model = NeuralNetwork([1, 2, 1])
    example = np.array([[10], [15]])
    print(model.predict(example))
    print(model.predict(example).shape)
