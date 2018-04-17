import gym

import numpy as np
from src.population.population import Population
from src.configs import *


class Game:
    # Read existing weights.
    def read_weights(self, weights_file_path):
        file = open(weights_file_path, 'r')

        weights_string = file.read().splitlines()

        file.close()

        weights = []
        for i in range(weights_string.__len__()):
            weights.append(np.float(weights_string[i]))

        return weights

    def train(self):
        # Creating environment
        env = gym.make('LunarLander-v2')

        # Population
        population = Population(POPULATION_SIZE)

        # Load weights with best score
        population.update_individuals_with_pre_existed_weights(self.read_weights(TOP_SCORE_WEIGHTS_LOAD_FILE_PATH))

        # Number of generations.
        for i in range(NUMBER_OF_GENERATIONS):

            # Iterate over all individuals of current generation, each individual plays the game.
            population.current_generation_best_fitness = -10000

            for individual_index in range(population.population_size):
                observation = env.reset()

                award = 0
                for t in range(PLAYING_TIME):
                    if i > 0:
                        env.render()

                    action = population.individuals[individual_index].get_action(observation, 0.5)
                    observation, reward, done, info = env.step(int(action))

                    award += reward

                    if done:
                        break

                population.individuals[individual_index].fitness = award

            population.update_current_generation_best_fitness()

            print("Generation #", i + 1, " , Max Score: ", population.current_generation_best_fitness)
            print("Population best individual output:", population.get_best_individual().brain.neural_network_output)

            # Reproduce by creating new generation with parents traits.
            population.reproduce()
            np.savetxt(TOP_SCORE_WEIGHTS_SAVE_FILE_PATH, population.get_best_individual().brain.weights_to_array(), fmt='%f')

        env.close()

        # population[4].brain.feed_forward([-0.9, -0.8, -0.3, -0.3, -0.9, -0.8, -0.9, -0.9])
        # neural_network.feed_forward([-0.9, -0.8, -0.3, -0.3, -0.9, -0.8, -0.9, -0.9])
        # print("output ", population[0].brain.neural_network_output)
