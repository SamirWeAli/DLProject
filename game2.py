import gym
import numpy as np
import neuralnetwork as nn
from individual import Individual
from population import Population
import math

from matplotlib import pyplot as plt
from random import randint
from statistics import median, mean

env = gym.make('LunarLander-v2')

# Individual playing time ( play for 100 steps if not died ).
playing_time = 10000

# population size
population_size = 20



# Population
population = Population(population_size)

# Read existing weights.
file = open('best_records/topscore.txt', 'r')
weights_string = file.read().splitlines()
file.close()
weights = []
for i in range(weights_string.__len__()):
    weights.append( np.float(weights_string[i]) )

# Load weights with best score
population.update_individuals_with_pre_existed_weights(weights)
# Number of generations.
for i in range(5000):
    # Iterate over all individuals of current generation, each individual plays the game.
    population.current_generation_best_fitness = -10000
    for individual_index in range(population.population_size):
        observation = env.reset()
        award = 0
        for t in range(playing_time):
            if i >0:
                env.render()
            action = population.individuals[individual_index].get_action(observation, 0.5)

            observation, reward, done, info = env.step(int(action))
            award += reward
            if done:
                break
        population.individuals[individual_index].fitness = award

    population.update_current_generation_best_fitness()
    print(" generation # ", i + 1, "max score : ", population.current_generation_best_fitness)
    print("Population best individual output", population.get_best_individual().brain.neural_network_output)

    # Reproduce by creating new generation with parents traits.
    population.reproduce()
    np.savetxt("topscore.txt",population.get_best_individual().brain.weights_to_array(), fmt='%f')


env.close()
#population[4].brain.feed_forward([-0.9, -0.8, -0.3, -0.3, -0.9, -0.8, -0.9, -0.9])
#neural_network.feed_forward([-0.9, -0.8, -0.3, -0.3, -0.9, -0.8, -0.9, -0.9])

#print("output ", population[0].brain.neural_network_output)