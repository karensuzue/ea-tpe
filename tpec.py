import numpy as np
import copy 
import ray
from ray import ObjectRef
from config import Config
from logger import Logger
from typing import List, Dict
from ea import EA
from tpe import TPE
from individual import Individual
from param_space import ModelParams
from utils import eval_factory, eval_final_factory

class TPEC:
    def __init__(self, config: Config, logger: Logger, param_space: ModelParams):
        self.config = config
        self.logger = logger
        self.param_space = param_space

        self.ea = EA(config, logger, param_space) 
        self.tpe = TPE(self.config.rng) # default gamma (splitting threshold)

        self.population: List[Individual] = []
        self.evaluated_individuals: List[Individual] = [] # archive of every individual evaluated so far

        # to keep current best Individuals
        self.best_performance = 1.0
        self.best_performers: List[Dict] = []

        # For debugging
        self.hard_eval_count = 0 # evaluations on the true objective
        self.soft_eval_count = 0 # evaluations on the surrogate/expected improvement
    
    def run(self, X_train, y_train, X_test, y_test):
        """ 
        Run TPEC with parallelized fitness evaluations using Ray. 

        Parameters:
            X_train, y_train: The training data.
            X_test, y_test: The testing data.
        """
        # Initialize Ray (multiprocessing)
        if not ray.is_initialized():
            ray.init(num_cpus = self.config.num_cpus, include_dashboard = True)

        # Initialize population with random organisms
        self.population = [Individual(self.param_space.generate_random_parameters()) for _ in range(self.config.pop_size)]

        # Make sure parameters align with scikit-learn's requirements before CV evaluations
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
            done, ray_evals = ray.wait(ray_evals, num_returns = 1)
            # Extract the only result (Tuple[float, int]) from the 'done' list
            performance, index = ray.get(done)[0] 
            self.population[index].set_performance(performance)
            if self.config.debug: self.hard_eval_count += 1

        # Remove individuals with positive performance
        self.remove_failed_individuals()
        # Update best performance and set of best performers
        self.process_population_for_best()
        print(f"Initial population size: {len(self.population)}")
        print(f"Best training performance so far: {self.best_performance}")

        # Append the population to the archive of all evaluated individuals
        self.evaluated_individuals += self.population
        assert(len(self.evaluated_individuals) == len(self.population))

        generations = (self.config.evaluations // self.config.pop_size) - 1
        for gen in range(generations):
            # Log population performance
            self.logger.log_generation(gen, self.config.pop_size * (gen + 1), self.population, "TPEC")

            # Can't fit KDEs over numeric parameters with value "None" (e.g. max_samples), set them to ~0
            for ind in self.evaluated_individuals:
                params = ind.get_params()
                for name in params:
                    if self.param_space.param_space[name]['type'] in ['int', 'float'] and params[name] is None:
                        params[name] = .0001
                ind.set_params(params)

            # Fit TPE to the archive of all observed individuals 
            self.tpe.fit(self.evaluated_individuals, self.param_space)

            # Fix the archive again to align with scikit-learn requirements
            for ind in self.evaluated_individuals:
                self.param_space.fix_parameters(ind.get_params())

            # To store the next population
            new_pop = []
            # List containing the expected improvement scores of the chosen offspring (1 per parent),
            # or essentially, members of the new population
            ei_all_parents: List[float] = [] 

            # Select enough parents to cover the entire population using tournament selection
            # These should already be fixed
            performances = np.array([ind.get_performance() for ind in self.population]) # from the current population
            parents = [self.ea.tournament_selection(self.population, performances) for _ in range(self.config.pop_size)]
            assert(len(parents) == self.config.pop_size)

            for parent in parents:
                # Each parent produces 'num_candidates' candidate offsprings (10 by default)
                candidates = [Individual(copy.deepcopy(parent.get_params())) for _ in range(self.config.num_candidates)]
                for child in candidates:
                    # Mutations occur in place, should also include fixing
                    self.param_space.mutate_parameters(child.get_params(), self.config.mut_rate)

                # Can't use TPE's suggest() on candidates if numeric parameters have value "None" (e.g. max_samples)
                for ind in candidates:
                    params = ind.get_params()
                    for name in params:
                        if self.param_space.param_space[name]['type'] in ['int', 'float'] and params[name] is None:
                            params[name] = .0001
                    ind.set_params(params)

                # Evaluate each candidate's expected improvement score, suggest one
                best_candidate, best_ei_scores, se_count = self.tpe.suggest(self.param_space, candidates, num_top_cand = 1) 
                if self.config.debug: self.soft_eval_count += se_count
                assert len(best_candidate) == 1

                # Make sure chosen candidate(s) align with scikit-learn's requirements before evaluation
                for ind in best_candidate: # 'best_candidate' should contain a single Individual
                    self.param_space.fix_parameters(ind.get_params()) # fixes in-place
                

                # Evaluate best candidate(s) on the true objective.
                # The system allows 'best_candidate' to contain more than one Individual (num_top_cand >= 1),
                # but in TPEC, this is always set to 1. 
                ray_child_evals: List[ObjectRef] = [
                    eval_factory.remote(self.config.model, ind.get_params(), X_train_ref, y_train_ref, self.config.seed, i)
                    for i, ind in enumerate(best_candidate) # should be 1 ind, by default
                ]
                while len(ray_child_evals) > 0:
                    done, ray_child_evals = ray.wait(ray_child_evals, num_returns = 1)
                    performance, index = ray.get(done)[0]
                    best_candidate[index].set_performance(performance)
                    if self.config.debug: self.hard_eval_count += 1

                # Append to new population
                new_pop += best_candidate # a bit sloppy, but works
                ei_all_parents += best_ei_scores.tolist() # this should be one score

            # Update population  
            self.population = new_pop

            self.remove_failed_individuals()
            self.process_population_for_best()
            print(f"Population size at gen {gen}: {len(self.population)}")
            print(f"Best training performance so far: {self.best_performance}")

            # Log per-iteration expected improvement statistics
            assert(len(ei_all_parents) == self.config.pop_size) 
            self.logger.log_ei(gen, self.config.pop_size * (gen + 1), ei_all_parents)

            # Append current population to archive of evaluated individuals 
            self.evaluated_individuals += self.population

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

        self.logger.log_generation(generations, self.config.pop_size * (generations + 1), self.population, "TPEC")
        self.logger.log_best(best_ind, self.config, "TPEC")
        self.logger.save(self.config, "TPEC")

        if self.config.debug:
            print(f"Hard evaluations: {self.hard_eval_count}")
            print(f"Soft evaluations {self.soft_eval_count}")

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
                self.best_performers.append(copy.deepcopy(ind.get_params()))

        assert len(self.best_performers) > 0, "No best performers found in the population."
        return