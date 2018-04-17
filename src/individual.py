import numpy as np

from src import neuralnetwork as nn


class Individual:

    def __init__(self, neural_network=None):
        if neural_network:
            self.brain = neural_network
        else:
            self.brain = nn.NeuralNetwork(8, 8, 4)

        self.fitness = 0

    # Get action based on input to the brain ( neural network ).
    def get_action(self, observation, threshold=0.5):

        self.brain.feed_forward(observation)
        """
        # Iterate over all outputs and threshold them.
        for i in range(self.brain.n_output_layer_nodes):
            if self.brain.neural_network_output[i] >= threshold:
                self.brain.neural_network_output[i] = 1
            else:
                self.brain.neural_network_output[i] = 0
        
        return self.brain.neural_network_output
        """

        # Get action based on heighest probability
        max_output = 0
        action = 0
        for i in range(self.brain.n_output_layer_nodes):
            if self.brain.neural_network_output[i] > max_output:
                action = i
                max_output = self.brain.neural_network_output[i]
        #print(" action ", action)
        return np.argmax(self.brain.neural_network_output)
