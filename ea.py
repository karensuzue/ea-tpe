import numpy as np
# import ray
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

    def make_offspring(self, population: List[Organism], num_offspring: int) -> List[Organism]:
        """ 
        Produce offspring asexually from a given population. 
        Parents are chosen using tournament selection.
        
        Parameters:
            population (List[Organism]):
            num_offspring (int): 
        """
        offspring = []
        # Assume organisms are already evaluated...
        fitnesses = np.array([org.get_fitness() for org in self.population])
        for _ in range(num_offspring): 
            # Randomly choose a tour_size number of indices
            indices = self.config.rng.choice(len(self.population), self.config.tour_size, replace=False)
            # Extract fitnesses at the chosen indices
            extracted_fitnesses = fitnesses[indices]
            # Get the position of the best (lowest) fitness in tournament (not population-based index)
            best_tour_idx = np.argmin(extracted_fitnesses)
            
            # Get the best individual among tournament contestants, the parent
            parent = population[indices[best_tour_idx]]

            # In-place mutation, deep copy required 
            child = copy.deepcopy(parent)
            self.param_space.mutate_parameters(child.get_genotype(), self.config.mut_rate)

            offspring.append(child)
        assert(len(offspring) == num_offspring)
        return offspring

    # Run "default" EA 
    # @ray.remote
    def run(self, X_train, y_train) -> None:
        # Initialize population with random organisms
        self.population = [Organism(self.param_space) for _ in range(self.config.pop_size)]

        for org in self.population:
            org.set_fitness(-1 * self.param_space.eval_parameters(org.get_genotype(), X_train, y_train))
            if self.config.debug: self.hard_eval_count += 1

        generations = (self.config.evaluations // self.config.pop_size) - 1
        for gen in range(generations):
            self.logger.log_generation(gen, self.config.pop_size * (gen + 1), self.population, "EA")

            self.population = self.make_offspring(self.population, self.config.pop_size)
            
            for org in self.population:
                org.set_fitness(-1 * self.param_space.eval_parameters(org.get_genotype(), X_train, y_train))
                if self.config.debug: self.hard_eval_count += 1

        # For the final generation
        self.population.sort(key=lambda o: o.get_fitness()) # ascending order
        best_org = self.population[0]
        self.param_space.fix_parameters(best_org.get_genotype())

        self.logger.log_generation(generations, self.config.pop_size * (generations + 1), self.population, "EA")
        self.logger.log_best(best_org, self.config, "EA")
        self.logger.save(self.config, "EA")

        if self.config.debug:
            print(f"Hard evaluations: {self.hard_eval_count}")