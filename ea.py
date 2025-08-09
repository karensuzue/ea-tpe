import numpy as np
import copy
import ray
from ray import ObjectRef
from config import Config
from logger import Logger
from individual import Individual
from param_space import ModelParams
from typeguard import typechecked
from typing import List, Tuple, Dict
from utils import ray_eval_factory, eval_final_factory
import copy as cp

@typechecked
class EA:
    """
    EA Solver.
    """
    def __init__(self, config: Config, logger: Logger, param_space: ModelParams):
        self.config = config
        self.logger = logger

        self.param_space = param_space

        self.population: List[Individual] = []

        # For debugging
        self.hard_eval_count = 0 # evaluations on the true objective

        # to keep current best Individuals
        self.best_performance = 1.0
        self.best_performers: List[Dict] = []

    def get_population(self) -> List[Individual]:
        return self.population

    def set_population(self, population: List[Individual]):
        self.population = population

    def tournament_selection(self, population: List[Individual], performances: np.ndarray) -> Individual:
        """
        Selects a single parent using tournament selection.

        Parameters:
            population (List[Individual]): A population of individuals to select from.
            performances (np.ndarray): A 1D array of performance values corresponding to the individuals in the population
                (Must have the same ordering). This serves to avoid redundant computations.
        Returns:
            Individual: a single parent individual.

        """
        assert(len(population) == len(performances))

        # Randomly choose a tour_size number of indices (determined by config)
        indices = self.config.rng.choice(len(population), self.config.tour_size, replace=False)
        # Extract performances at the chosen indices
        extracted_performances = performances[indices]
        # Get the position of the best (lowest) performance in the tournament (not population-based index)
        best_tour_idx = np.argmin(extracted_performances)
        # Get the best individual among tournament contestants, the parent
        parent = population[indices[best_tour_idx]]
        return parent

    def make_offspring(self, population: List[Individual], num_offspring: int) -> List[Individual]:
        """
        Produce freshly made offspring asexually from a given population.
        Parents are chosen using tournament selection.

        Parameters:
            population (List[Individual]): A population of individuals to select from.
            num_offspring (int): The number of offspring to produce.
        """
        offspring_params = []
        # Assume individuals are already evaluated...
        performances = np.array([ind.get_performance() for ind in self.population])
        for _ in range(num_offspring):
            # Select a single parent using tournament selection
            parent = self.tournament_selection(population, performances)

            # In-place mutation with fixing, deep copy required
            child = copy.deepcopy(parent.get_params())
            self.param_space.mutate_parameters(child, self.config.mut_rate)
            offspring_params.append(child)

        assert(len(offspring_params) == num_offspring)
        return [Individual(params) for params in offspring_params]

    def run(self, X_train, y_train, X_test, y_test) -> None:
        """
        Run the default EA with parallelized fitness evaluations using Ray.

        Parameters:
            X_train, y_train: The training data.
            X_test, y_test: The testing data.
        """
        # Initialize Ray (multiprocessing)
        if not ray.is_initialized():
            ray.init(num_cpus = self.config.num_cpus, include_dashboard = True)

        # Initialize population with random individuals
        self.population = [Individual(self.param_space.generate_random_parameters()) for _ in range(self.config.pop_size)]

        # Create Ray ObjectRefs to efficiently share across multiple Ray tasks
        X_train_ref = ray.put(X_train)
        y_train_ref = ray.put(y_train)

        # Evaluate population with Ray; 1 remote task per individual or set of hyperparameters
        ray_evals: List[ObjectRef] = [
            ray_eval_factory.remote(self.config.model, ind.get_params(), X_train_ref, y_train_ref, self.config.seed, i)
            for i, ind in enumerate(self.population) # each ObjectRef leads to a Tuple[float, int]
        ]
        # Process results as they come in
        while len(ray_evals) > 0:
            done, ray_evals = ray.wait(ray_evals, num_returns = 1)
            performance, index = ray.get(done)[0]
            self.population[index].set_performance(performance)
            if self.config.debug: self.hard_eval_count += 1

        # Remove individuals with positive performance
        self.remove_failed_individuals()
        self.process_population_for_best()
        print(f"Initial population size: {len(self.population)}")
        print(f"Best training performance so far: {self.best_performance}")

        # Start evolution
        generations = (self.config.evaluations // self.config.pop_size) - 1
        for gen in range(generations):
            self.logger.log_generation(gen, self.config.pop_size * (gen + 1), self.population, "EA")

            # Produce offspring (parents may be carried over unchanged, we consider them to be new offspring regardless)
            self.population = self.make_offspring(self.population, self.config.pop_size)

            # Evaluate offspring with Ray
            ray_child_evals: List[ObjectRef] = [
                ray_eval_factory.remote(self.config.model, ind.get_params(), X_train_ref, y_train_ref, self.config.seed, i)
                for i, ind in enumerate(self.population)
            ]
            while len(ray_child_evals) > 0:
                # ray_child_evals shrinks as it gets updated
                done, ray_child_evals = ray.wait(ray_child_evals, num_returns = 1)
                performance, index = ray.get(done)[0]
                self.population[index].set_performance(performance)
                if self.config.debug: self.hard_eval_count += 1

            # remove individuals with positive performance
            self.remove_failed_individuals()
            self.process_population_for_best()
            print(f"Population size at gen {gen}: {len(self.population)}")
            print(f"Best training performance so far: {self.best_performance}")

        # randomly select one of the tied best individuals
        assert len(self.best_performers) > 0, "No best performers found in the population."
        best_ind_params = self.config.rng.choice(self.best_performers)

        # Final scores
        train_accuracy, test_accuracy = eval_final_factory(self.config.model, best_ind_params,
                                                           X_train, y_train, X_test, y_test, self.config.seed)
        best_ind = Individual(params=best_ind_params,
                              performance=self.best_performance,
                              train_score=train_accuracy,
                              test_score=test_accuracy)

        self.logger.log_generation(generations, self.config.pop_size * (generations + 1), self.population, "EA")
        self.logger.log_best(best_ind, self.config, "EA")
        self.logger.save(self.config, "EA")

        if self.config.debug:
            print(f"Hard evaluations: {self.hard_eval_count}")

        ray.shutdown()

    # remove any individuals wiht a positive performance
    def remove_failed_individuals(self) -> None:
        """
        Removes individuals with a positive performance from the population.
        This is useful for ensuring that only individuals with negative performance are considered.
        A positive performance indicates that the individual failed during evaluation and is not suitable for selection.
        """
        self.population = [ind for ind in self.population if ind.get_performance() <= 0.0]
        if self.config.debug:
            print(f"Removed individuals with positive performance, new population size: {len(self.population)}")
        return

    # return the best training performance from the population
    def get_best_performance(self) -> float:
        """
        Returns the best training performance from the population.
        """
        if not self.population:
            raise ValueError("Population is empty, cannot get best training performance.")
        return min([ind.get_performance() for ind in self.population])

    # process the current population and update self.best_performers and self.best_performance
    def process_population_for_best(self) -> None:
        """
        Processes the current population and updates self.best_performers and self.best_performance.
        """
        current_best = self.get_best_performance()

        # check if we have found a better performance
        if current_best < self.best_performance:
            self.best_performance = current_best
            self.best_performers = []

        # add all individuals with the current best performance to the best performers
        for ind in self.population:
            if ind.get_performance() == self.best_performance:
                self.best_performers.append(cp.deepcopy(ind.get_params()))

        assert len(self.best_performers) > 0, "No best performers found in the population."
        return