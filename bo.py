import numpy as np
import copy
import ray
from ray import ObjectRef
from config import Config
from logger import Logger
from param_space import ModelParams
from individual import Individual
from surrogate import Surrogate
from tpe import TPE
from typeguard import typechecked
from typing import Tuple, Dict, List, Literal
from utils import eval_factory, eval_final_factory, bo_eval_parameters_RF

@typechecked
class BO:
    """
    This class implements a Bayesian Optimizer and is capable of supporting different surrogate models.
    The core steps to BO are:
        1. Observe a small number of points from the search space.
        2. Fit a surrogate model to the observed data.
        3. Using an acquisition function to select the next candidate(s).
        4. Evaluate the selected candidate(s) on the objective function.
        5. Add new data point(s) to the set of observations, and refit the surrogate model.

    In our system, the size of the initial observed set ('samples') is synonymous to EA's population size.
    """
    def __init__(self, config: Config, logger: Logger, param_space: ModelParams, surrogate: Literal["TPE", "GP"],
                 num_top_cand: int = 1):
        """
        Parameters:
            config (Config):
            logger (Logger):
            param_space (ModelParams):
            surrogate (Literal["TPE", "GP"]):
            num_top_cand (int): The number of selected candidate(s) to observe and add to the set of observations.
        """
        self.config = config
        self.logger = logger
        self.param_space = param_space
        self.surrogate_type = surrogate
        self.num_top_cand = num_top_cand

        if self.surrogate_type == "TPE":
            self.surrogate: Surrogate = TPE(self.config.rng)

        self.samples: List[Individual] = [] # This stores the observed samples

        # to keep current best (tied) hyperparameter sets
        self.best_performers: List[Dict] = []
        self.best_performance = 1.0

        # For debugging
        self.hard_eval_count = 0
        self.soft_eval_count = 0

    def get_samples(self) -> List[Individual]:
        return self.samples

    def set_samples(self, samples: List[Individual]):
        self.samples = samples

    def run(self, X_train, y_train, X_test, y_test):
        """
        Run BO with parallelized fitness evaluations using Ray.

        Parameters:
            X_train, y_train: The training data.
            X_test, y_test: The testing data.
        """
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(num_cpus = self.config.num_cpus, include_dashboard = True)

        # Initialize observed sample set, its size is determined by the 'pop_size' parameter in the Config
        self.samples = [Individual(self.param_space.generate_random_parameters()) for _ in range(self.config.pop_size)]

        # Create Ray ObjectRefs to efficiently share across multiple Ray tasks
        X_train_ref = ray.put(X_train)
        y_train_ref = ray.put(y_train)

        # Evaluate initial samples with Ray
        ray_evals: List[ObjectRef] = [
            eval_factory.remote(self.config.model, ind.get_params(), X_train_ref, y_train_ref, self.config.seed, i)
            for i, ind in enumerate(self.samples) # each ObjectRef leads to a Tuple[float, int]
        ]
        # Process results as they come in
        while len(ray_evals) > 0:
            done, ray_evals = ray.wait(ray_evals, num_returns = 1)
            # Extract the only result (Tuple[float, int]) from the 'done' list
            performance, index = ray.get(done)[0]
            self.samples[index].set_performance(performance)
            if self.config.debug: self.hard_eval_count += 1

        # Remove individuals with positive performance
        self.remove_failed_individuals()
        # Update best performance and set of best performers in the current sample set
        self.process_samples_for_best()
        print(f"Initial sample size: {len(self.samples)}")
        print(f"Best training performance so far: {self.best_performance}")

        # Can't fit KDEs over numeric parameters with value "None" (e.g. max_samples), set them to a small value
        for ind in self.samples:
            params = ind.get_params()
            for name in params:
                if self.param_space.param_space[name]['type'] in ['float'] and params[name] is None:
                    params[name] = 1.0e-16
                if self.param_space.param_space[name]['type'] in ['int'] and params[name] is None:
                    params[name] = 0

        generations = (self.config.evaluations - self.config.pop_size) // self.num_top_cand
        for gen in range(generations):
            # Log best, average, and median objective values in the current sample set
            self.logger.log_generation(gen, self.config.pop_size + gen * self.num_top_cand,
                                       self.samples, f"{self.surrogate_type}BO")

            # Fit the surrogate to the observed data
            self.surrogate.fit(self.samples, self.param_space, self.config.rng)

            # Sample enough candidates from the surrogate to keep the number of 'soft' evaluations consistent between BO and TPEC
            candidates = self.surrogate.sample(self.config.num_candidates * self.num_top_cand, self.param_space, self.config.rng)

            # Select the top candidate(s) for evaluation on the true objective
            best_candidates, ei_scores, soft_eval_count = self.surrogate.suggest(self.param_space, candidates, self.num_top_cand)
            if self.config.debug: self.soft_eval_count += soft_eval_count

            # Log per-iteration expected improvement statistics (only from the chosen candidates)
            self.logger.log_ei(gen, self.config.pop_size + gen * self.num_top_cand, ei_scores)

            # Make sure chosen candidate(s) align with scikit-learn's requirements before evaluation
            for ind in best_candidates: # 'best_candidates' may contain more than 1 Individual
                self.param_space.fix_parameters(ind.get_params()) # fixes in-place

            for ind in best_candidates:
                ind.set_performance(bo_eval_parameters_RF(ind.get_params(),
                                                          X_train,
                                                          y_train,
                                                          self.config.seed,
                                                          self.config.num_cpus))
                self.hard_eval_count += 1

            # must update best candidates for the surrogate
            for ind in best_candidates:
                params = ind.get_params()
                for name in params:
                    if self.param_space.param_space[name]['type'] in ['float'] and params[name] is None:
                        params[name] = 1.0e-16
                    if self.param_space.param_space[name]['type'] in ['int'] and params[name] is None:
                        params[name] = 0

            # Update sample set
            self.samples += best_candidates

            # remove individuals with positive performance
            self.remove_failed_individuals()
            # Update best performance and set of best performers
            self.process_samples_for_best()
            print(f"Samples size at gen {gen}: {len(self.samples)}")
            print(f"Best training performance so far: {self.best_performance}")

        # randomly select one of the tied best individuals
        assert len(self.best_performers) > 0, "No best performers found in the population."
        best_ind_params = self.config.rng.choice(self.best_performers)

        # revert samples for best_ind
        self.param_space.fix_parameters(best_ind_params)

        # Final scores
        train_accuracy, test_accuracy = eval_final_factory(self.config.model, best_ind_params,
                                                           X_train, y_train, X_test, y_test, self.config.seed)
        best_ind = Individual(params=best_ind_params,
                              performance=self.best_performance,
                              train_score=train_accuracy,
                              test_score=test_accuracy)



        # Log best, average, and median objective values in the final sample set
        self.logger.log_generation(generations, self.config.pop_size + generations * self.num_top_cand,
                                   self.samples, f"{self.surrogate_type}BO")
        # Log the best observed hyperparameter configuration across all iterations
        self.logger.log_best(best_ind, self.config, f"{self.surrogate_type}BO")
        self.logger.save(self.config, f"{self.surrogate_type}BO")

        if self.config.debug:
            print(f"Hard evaluations: {self.hard_eval_count}")
            print(f"Soft evaluations {self.soft_eval_count}")
            assert(len(self.samples) == self.config.evaluations)

        ray.shutdown()


    # remove any individuals wiht a positive performance
    def remove_failed_individuals(self) -> None:
        """
        Removes individuals with a positive performance from the population.
        This is useful for ensuring that only individuals with negative performance are considered.
        A positive performance indicates that the individual failed during evaluation and is not suitable for selection.
        """
        self.samples = [ind for ind in self.samples if ind.get_performance() <= 0.0]
        if self.config.debug:
            print(f"Removed individuals with positive performance, new population size: {len(self.samples)}")
        return

    # return the best training performance from the population
    def get_best_performance(self) -> float:
        """
        Returns the best training performance from the population.
        """
        if not self.samples:
            raise ValueError("Population is empty, cannot get best training performance.")
        return min([ind.get_performance() for ind in self.samples])

    # process the current sample set and update self.best_performers and self.best_performance
    def process_samples_for_best(self) -> None:
        """
        Processes the current population and updates self.best_performers and self.best_performance.
        """
        current_best = self.get_best_performance()

        # check if we have found a better performance
        if current_best < self.best_performance:
            self.best_performance = current_best
            self.best_performers = []

        # add all individuals with the current best performance to the best performers
        for ind in self.samples:
            if ind.get_performance() == self.best_performance:
                self.best_performers.append(copy.deepcopy(ind.get_params()))

        assert len(self.best_performers) > 0, "No best performers found in the population."
        return