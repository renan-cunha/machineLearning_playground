import numpy as np
from typing import List
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod
from typing import Tuple, List
from layers import LinearLayer, SigmoidLayer, SqrErrorLayer


class NeuralNetwork:

    def __init__(self, num_neurons: List[int]):
        self.learning_rate = 0.1
        self.training_iters = 1
        self.layers = []
        for index, neuron in enumerate(num_neurons[:-2]):
            print(neuron, num_neurons[index+1])
            layer = LinearLayer((neuron, num_neurons[index+1]))
            self.layers.append(layer)
            layer = SigmoidLayer()
            self.layers.append(layer)
        layer = LinearLayer((num_neurons[-2], num_neurons[-1]))
        self.layers.append(layer)
        self.loss_layer = None
    
    def backward_propagation(self, output: np.ndarray) -> None:
        grad_input = self.loss_layer.backward(np.ones_like(output)) 
        for layer_index in range(len(self.layers)-1, -1, -1):
            grad_input = self.layers[layer_index].backward(grad_input)

    def update_params(self, learning_rate: float) -> None:
        for layer_index in range(len(self.layers)):
            layer = self.layers[layer_index]
            if type(layer) == LinearLayer:
                self.layers[layer_index].weights -= learning_rate*layer.grad_weights
                self.layers[layer_index].biases -= learning_rate*layer.grad_biases
                

    def fit(self, x: np.ndarray, y: np.ndarray,
            learning_rate: float, training_iters: int) -> np.array:
        self.loss_layer = SqrErrorLayer(y)

        losses = np.empty(self.training_iters)
        for i in range(training_iters):

            y_pred = self.predict(x)
            error = self.loss_layer.forward(y_pred)
            mean_error = np.mean(error)
            self.backward_propagation(y_pred)
                   
            print(f"EPOCH {i}")
            print(f"MSE Error: {mean_error}")
            self.update_params(learning_rate)

        return losses

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

if __name__ == "__main__":
    model = NeuralNetwork([1, 10, 1])
    x = np.random.randn(10).reshape(-1, 1)
    y = 5 + 3*x + np.random.normal(0, 3)
    print(x.shape)
    print(y.shape)
    model.fit(x, y, training_iters=1000, learning_rate=0.001)
