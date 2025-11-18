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
from utils import evaluation, eval_final_factory, remove_failed_individuals, process_population_for_best
import copy as cp
from sklearn.model_selection import KFold


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

        # Randomly choose a tour_size number of population indices (determined by config)
        indices = self.config.rng.choice(len(population), self.config.tour_size, replace=False)
        # Extract performances at the chosen indices
        extracted_performances = performances[indices]
        # Get the position of the best (lowest) performance in the tournament (not population-based index)
        best_tour_idx = np.argmin(extracted_performances)
        # Find the population index of all individuals in tournament with the best performance (in case of ties)
        best_indices = [i for i, perf in zip(indices, extracted_performances) if perf == extracted_performances[best_tour_idx]]
        # Randomly select one of the tied best individuals (population-based index)
        best_tour_idx = self.config.rng.choice(best_indices)
        # Get the best individual among tournament contestants, the parent
        parent = population[best_tour_idx]
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
            child_params = copy.deepcopy(parent.get_params())
            self.param_space.mutate_parameters(child_params, self.config.mut_rate, self.config.mut_var)
            offspring_params.append(child_params)

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
            ray.init(num_cpus = self.config.num_cpus, include_dashboard = False)

        # Initialize population with random individuals
        self.population = [Individual(self.param_space.generate_random_parameters()) for _ in range(self.config.pop_size)]

        # Create Ray ObjectRefs to efficiently share across multiple Ray tasks
        X_train_ref = ray.put(X_train)
        y_train_ref = ray.put(y_train)

        # create cv splits for cross-validation
        cv = KFold(n_splits=self.config.cv_k, shuffle=True, random_state=self.config.seed)
        splits = list(cv.split(X_train, y_train))

        # Evaluate initial population with Ray
        evaluation(self.population, X_train_ref, y_train_ref, splits, self.config.model, self.config.seed)
        if self.config.debug: self.hard_eval_count += len(self.population)

        # Remove individuals with positive performance
        self.population = remove_failed_individuals(self.population, self.config)
        self.best_performance, self.best_performers = process_population_for_best(self.population, self.best_performance, self.best_performers)
        print(f"Initial population size: {len(self.population)}", flush=True)
        print(f"Best training performance so far: {self.best_performance}", flush=True)

        # Start evolution
        generations = (self.config.evaluations // self.config.pop_size) - 1
        for gen in range(generations):
            self.logger.log_generation(gen, self.config.pop_size * (gen + 1), self.population, "EA")

            # Produce offspring (parents may be carried over unchanged, we consider them to be new offspring regardless)
            self.population = self.make_offspring(self.population, self.config.pop_size)

            # Evaluate offspring with Ray
            evaluation(self.population, X_train_ref, y_train_ref, splits, self.config.model, self.config.seed)
            if self.config.debug: self.hard_eval_count += len(self.population)

            # remove individuals with positive performance
            self.population = remove_failed_individuals(self.population, self.config)
            self.best_performance, self.best_performers = process_population_for_best(self.population, self.best_performance, self.best_performers)
            print(f"Population size at gen {gen}: {len(self.population)}", flush=True)
            print(f"Best training performance so far: {self.best_performance}", flush=True)

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
            print(f"Hard evaluations: {self.hard_eval_count}", flush=True)

        ray.shutdown()