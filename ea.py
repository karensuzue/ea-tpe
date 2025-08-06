import numpy as np
import copy
import ray
from ray import ObjectRef
from config import Config
from logger import Logger
from individual import Individual
from param_space import ModelParams
from typeguard import typechecked
from typing import List, Tuple
from utils import eval_factory, eval_final_factory

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
        Produce offspring asexually from a given population. 
        Parents are chosen using tournament selection.
        
        Parameters:
            population (List[Individual]): A population of individuals to select from.
            num_offspring (int): The number of offspring to produce.
        """
        offspring = []
        # Assume individuals are already evaluated...
        performances = np.array([ind.get_performance() for ind in self.population])
        for _ in range(num_offspring): 
            # Select a single parent using tournament selection
            parent = self.tournament_selection(population, performances)

            # In-place mutation with fixing, deep copy required 
            child = copy.deepcopy(parent)
            self.param_space.mutate_parameters(child.get_params(), self.config.mut_rate)

            offspring.append(child)
        
        assert(len(offspring) == num_offspring)
        return offspring

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

        # Make sure parameters align with scikit-learn's requirements
        for ind in self.population:
            self.param_space.fix_parameters(ind.get_params()) # fixes in-place 

        # Create Ray ObjectRefs to efficiently share across multiple Ray tasks 
        X_train_ref = ray.put(X_train)
        y_train_ref = ray.put(y_train)

        # Evaluate population with Ray; 1 remote task per individual or set of hyperparameters
        ray_evals: List[ObjectRef] = [ 
            eval_factory.remote(self.config.model, ind.get_params(), X_train_ref, y_train_ref, self.config.seed, i)
            for i, ind in enumerate(self.population) # each ObjectRef leads to a Tuple[float, int]
        ]
        # Process results as they come in
        while len(ray_evals) > 0:
            # ray.wait() returns a list of ObjectRefs for completed results & a list of ObjectRefs for remaining jobs. 
            # We remove the finished jobs from ray_evals by updating it to be the list of remaining jobs
            # Here we set 'num_returns' to 1, so it returns one finished reference at a time
            done, ray_evals = ray.wait(ray_evals, num_returns = 1)
            # Extract the only result (Tuple[float, int]) from the 'done' list
            performance, index = ray.get(done)[0] 
            self.population[index].set_performance(performance)
            if self.config.debug: self.hard_eval_count += 1
        
        # Start evolution
        generations = (self.config.evaluations // self.config.pop_size) - 1
        for gen in range(generations):
            self.logger.log_generation(gen, self.config.pop_size * (gen + 1), self.population, "EA")

            # Produce offspring (parents may be carried over unchanged, we consider them to be new offspring regardless)
            self.population = self.make_offspring(self.population, self.config.pop_size)
            
            # Evaluate offspring with Ray
            ray_child_evals: List[ObjectRef] = [ 
                eval_factory.remote(self.config.model, ind.get_params(), X_train_ref, y_train_ref, self.config.seed, i)
                for i, ind in enumerate(self.population)
            ]
            while len(ray_child_evals) > 0:
                # ray_child_evals shrinks as it gets updated
                done, ray_child_evals = ray.wait(ray_child_evals, num_returns = 1)
                performance, index = ray.get(done)[0]
                self.population[index].set_performance(performance)
                if self.config.debug: self.hard_eval_count += 1

        # Get all individuals tied for best performance, randomly select one of the tied best individuals
        self.population.sort(key=lambda o: o.get_performance()) # ascending order
        best_performance = self.population[0].get_performance()
        best_inds = [ind for ind in self.population if ind.get_performance() == best_performance]
        best_ind = self.config.rng.choice(best_inds)

        # Final scores
        train_accuracy, test_accuracy = eval_final_factory(self.config.model, best_ind.get_params(),
                                                           X_train, y_train, X_test, y_test, self.config.seed)
        best_ind.set_train_score(train_accuracy)
        best_ind.set_test_score(test_accuracy)

        self.logger.log_generation(generations, self.config.pop_size * (generations + 1), self.population, "EA")
        self.logger.log_best(best_ind, self.config, "EA")
        self.logger.save(self.config, "EA")

        if self.config.debug:
            print(f"Hard evaluations: {self.hard_eval_count}")
        
        ray.shutdown()
