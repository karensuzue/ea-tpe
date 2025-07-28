import numpy as np
import random
import copy
from config import Config
from logger import Logger
from organism import Organism
from param_space import ModelParams
from typeguard import typechecked
from typing import List

@typechecked
class EA:
    """
    EA Solver.
    """
    def __init__(self, config: Config, logger: Logger, param_space: ModelParams):
        self.config = config
        self.logger = logger

        self.param_space = param_space
        
        self.population: List[Organism] = []

        # For debugging
        self.hard_eval_count = 0 # evaluations on the true objective

    def get_population(self) -> List[Organism]:
        return self.population
    
    def set_population(self, population: List[Organism]):
        self.population = population

    def select_parents(self, num_parents: int) -> List[Organism]:
        """ 
        Tournament selection.
        Currently mutation only. 
        In the future, reference Starbase's 'variation_order' function and 
        calculate the number of parents needed to produce 'pop_size' offspring.
        """
        parents = []
        # Assume organisms are evaluated...
        fitnesses = np.array([org.get_fitness() for org in self.population])
        for _ in range(num_parents): 
            # Randomly choose a tour_size number of indices
            indices = self.config.rng.choice(len(self.population), self.config.tour_size, replace=False)
            # Extract fitnesses at the chosen indices
            extracted_fitnesses = fitnesses[indices]
            # Get the position of the best (lowest) fitness in tournament (not population-based index)
            best_tour_idx = np.argmin(extracted_fitnesses)
            best_idx = indices[best_tour_idx]
            parents.append(self.population[best_idx])
        return parents

    def make_offspring(self, parents: List[Organism]) -> List[Organism]:
        offspring = []
        for p in parents:
            child = copy.deepcopy(p)
            self.param_space.mutate_parameters(child.get_genotype(), self.config.mut_rate)
            offspring.append(child)
        return offspring

    # Run "default" EA 
    def run(self, X_train, y_train) -> None:
        # Initialize population with random organisms
        self.population = [Organism(self.param_space) for _ in range(self.config.pop_size)]

        for org in self.population:
            org.set_fitness(self.param_space.eval_parameters(org.get_genotype(), X_train, y_train))
            if self.config.debug: self.hard_eval_count += 1

        generations = (self.config.evaluations // self.config.pop_size) - 1
        for gen in range(generations):
            self.logger.log_generation(gen, self.config.pop_size * (gen + 1), self.population, "EA")

            # Replace the entire population with offspring
            parents = self.select_parents(self.config.pop_size)
            self.population = self.make_offspring(parents)
            
            for org in self.population:
                org.set_fitness(self.param_space.eval_parameters(org.get_genotype(), X_train, y_train))
                if self.config.debug: self.hard_eval_count += 1


        # For the final generation
        self.logger.log_generation(generations, self.config.pop_size * (generations + 1), self.population, "EA")
        self.logger.log_best(self.population, self.config, "EA")
        self.logger.save(self.config, "EA")

        if self.debug:
            print(f"Hard evaluations: {self.hard_eval_count}")