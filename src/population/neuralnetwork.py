import numpy as np
from src.configs import *

class Math:
    @staticmethod
    def reLu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


# Neural network class, hidden layers activation function ->ReLu, output activation function ->sigmoid
class NeuralNetwork(Math):
    def __init__(self, number_of_input_nodes, number_of_hidden_nodes, number_of_output_nodes):
        self.n_input_layer_nodes = number_of_input_nodes
        self.n_hidden_layer_nodes = number_of_hidden_nodes
        self.n_output_layer_nodes = number_of_output_nodes

        # Matrix (hidden_nodes(rows) x input_nodes(cols))
        self.input_layer_weights = np.random.uniform(INPUT_LAYER_WEIGHTS_RANGE[0], INPUT_LAYER_WEIGHTS_RANGE[1], (
            self.n_hidden_layer_nodes, self.n_input_layer_nodes)) / self.n_input_layer_nodes  # TODO @Ali ASK?

        self.input_layer_bias = np.random.uniform(INPUT_LAYER_BIAS_RANGE[0], INPUT_LAYER_BIAS_RANGE[1], (1, self.n_hidden_layer_nodes))

        # Matrix(output_nodes(rows) x hidden_nodes(cols))
        self.hidden_layer_weights = np.random.uniform(HIDDEN_LAYER_WEIGHTS_RANGE[0], HIDDEN_LAYER_WEIGHTS_RANGE[1], (
            self.n_output_layer_nodes, self.n_hidden_layer_nodes)) / self.n_hidden_layer_nodes

        self.neural_network_output = []

    def feed_forward(self, input):
        output_from_hidden_layer = (np.dot(self.input_layer_weights, input) + self.input_layer_bias)

        # Reshape -1 because it is array of arrays, we need it one array of size 1*n.
        self.neural_network_output = self.softmax(
            np.dot(self.hidden_layer_weights, output_from_hidden_layer.reshape(-1)))

    # Convert weights and bias term into 1d array ( to act as dna ).
    def weights_to_array(self):
        weights_and_biases = np.concatenate((self.input_layer_weights.reshape(-1),
                                             self.input_layer_bias.reshape(-1),
                                             self.hidden_layer_weights.reshape(-1)))
        return weights_and_biases

    # Convert array of weights, bias, hidden layer weights to 2d matrices of input,hidden layer weights and bias term
    def array_to_weights(self, array):
        array = np.asarray(array)

        # Positions in array.
        input_weights_start = 0
        input_weights_end = self.n_input_layer_nodes * self.n_hidden_layer_nodes

        input_bias_start = input_weights_end
        input_bias_end = input_bias_start + self.n_hidden_layer_nodes

        hidden_layer_start = input_bias_end
        hidden_layer_end = hidden_layer_start + self.n_hidden_layer_nodes * self.n_output_layer_nodes

        self.input_layer_weights = array[input_weights_start:input_weights_end].reshape(self.n_hidden_layer_nodes,
                                                                                        self.n_input_layer_nodes)

        self.input_layer_bias = array[input_bias_start:input_bias_end].reshape(1, self.n_hidden_layer_nodes)

        self.hidden_layer_weights = array[hidden_layer_start:hidden_layer_end].reshape(self.n_output_layer_nodes,
                                                                                       self.n_hidden_layer_nodes)
