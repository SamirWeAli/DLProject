from individual import Individual

import numpy as np
from copy import copy
class Population:

    def __init__(self, pop_size):
        self.first_time_debug = True

        self.mutation_rate = 0.0001
        self.population_size = pop_size
        self.individuals = []
        self.current_generation_best_fitness = -1000
        # Set initial population individuals with random weights.
        for i in range(self.population_size):
            self.individuals.append(Individual())

    def update_current_generation_best_fitness(self):

        for i in range(self.population_size):
            self.current_generation_best_fitness = np.maximum(self.current_generation_best_fitness,
                                                          int(self.individuals[i].fitness))

    # Get best individual.
    def get_best_individual(self):
        best_individual = self.individuals[0]

        for i in range(1,self.population_size):
            if self.individuals[i].fitness > best_individual.fitness:
                best_individual = self.individuals[i]

        return best_individual


    # Return worst individual in population
    def get_worst_individual(self):
        worst_individual = self.individuals[0]

        for i in range(1, self.population_size):
            if self.individuals[i].fitness < worst_individual.fitness:
                worst_individual = self.individuals[i]

        return worst_individual

    # Create new offsprings based upon natural selection.
    def reproduce(self):
        if self.first_time_debug:
            self.first_time_debug = False
            np.savetxt("initweights.txt", self.get_best_individual().brain.weights_to_array(), fmt='%f')

        # Create mating pool based on fitness of every individual, higher fitness means  higher selection probability
        mating_pool = []

        # Get minimum fitness to add it to all individuals fitness to avoid negative.
        min_fitness = self.get_worst_individual().fitness
        sorted(self.individuals, key=lambda individual: individual.fitness, reverse=True)

        # Number of occurrences in mating_pool is based on fitness.
        # Iterate over all individuals each individual will have n = fintness positions in mating pool.
        for i in range(self.population_size):
            for j in range(int(self.individuals[i].fitness + (-min_fitness+5))):
                mating_pool.append(i)

        # Create new temp individuals ( population).
        # Temp individuals weights  are to be updated in crossover and mutation.
        temp_individuals = []
        for i in range(self.population_size):
            temp_individuals.append(Individual())

        # For  n = population size , select two parents  make crossover process and return new weights for offspring.
        for i in range(self.population_size):

            # Select two random parents.
            # Select index from mating pool.
            indx_A = mating_pool[np.random.randint(len(mating_pool))]
            indx_B = mating_pool[np.random.randint(len(mating_pool))]

            parent_A = self.individuals[indx_A]
            parent_B = self.individuals[indx_B]

            offspring_DNA = self.cross_over(parent_A, parent_B)

            # Mutate the offspring DNA.
            # mutated_offspring_DNA = self.mutate(offspring_DNA.copy())
            mutated_offspring_DNA = self.mutate(offspring_DNA.copy())

            # Set weights with DNA.
            temp_individuals[i].brain.array_to_weights(mutated_offspring_DNA)

        # Set current individuals to new individuals.
        self.individuals = temp_individuals.copy()


    def cross_over(self, parent_a, parent_b):


        # Cross over between DNA's ( weights ).
        parent_a_weights_array = parent_a.brain.weights_to_array()
        parent_b_weights_array = parent_b.brain.weights_to_array()

        # 0 to midpoint from  parent_A DNA , midpoint to DNA end from parent_B
        midpoint = np.random.randint(0, parent_a_weights_array.size)


        # Offspring DNA
        offspring_DNA = []

        # Set offspring DNA  from parents by midpoint
        for i in range (len(parent_a_weights_array)):
            if i < midpoint:
                offspring_DNA.append(parent_a_weights_array[i])
            else:
                offspring_DNA.append(parent_b_weights_array[i])

        return offspring_DNA

    def mutate(self, offspring_DNA):
        mutated_DNA = offspring_DNA.copy()
        for i in range(len(offspring_DNA)):
            # Probability to mutate
            mutation_prob = np.random.rand(1)
            if(mutation_prob <= self.mutation_rate):
                mutated_DNA[i] = np.random.uniform(-0.1 , 0.1)

        return offspring_DNA

    # Update individuals with pre-existed weights.
    def update_individuals_with_pre_existed_weights(self,weights):
        for i in range(self.population_size):
            self.individuals[i].brain.array_to_weights(weights)
        print(weights)
