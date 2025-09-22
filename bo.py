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
from typing import Dict, List, Literal
from utils import eval_final_factory, remove_failed_individuals, process_population_for_best, evaluation
from sklearn.model_selection import KFold

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
            self.surrogate: Surrogate = TPE()

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
            ray.init(num_cpus = self.config.num_cpus, include_dashboard = False)

        # Initialize observed sample set, its size is determined by the 'pop_size' parameter in the Config
        self.samples = [Individual(self.param_space.generate_random_parameters()) for _ in range(self.config.pop_size)]

        # Create Ray ObjectRefs to efficiently share across multiple Ray tasks
        X_train_ref = ray.put(X_train)
        y_train_ref = ray.put(y_train)

        # create cv splits for cross-validation
        self.cv = KFold(n_splits=self.config.cv_k, shuffle=True, random_state=self.config.seed)
        self.splits = list(self.cv.split(X_train, y_train))

        # Evaluate initial population with Ray
        evaluation(self.samples, X_train_ref, y_train_ref, self.splits, self.config.model, self.config.seed)
        if self.config.debug: self.hard_eval_count += len(self.samples)

        # Remove individuals with positive performance in sample set
        self.samples = remove_failed_individuals(self.samples, self.config)
        # Update best performance and set of best performers in the current sample set
        self.best_performance, self.best_performers = process_population_for_best(self.samples, self.best_performance, self.best_performers)
        print(f"Initial sample size: {len(self.samples)}", flush=True)
        print(f"Best training performance so far: {self.best_performance}", flush=True)

        # TPE cannot be fitted over numeric parameters with the value "None" (e.g., max_samples),
        # so we create a deep copy of the original sample set and assign such parameters a value close to 0.
        # A deep copy is used to preserve the performance of the original individuals,
        # which is necessary for fitting a TPE over the parameter space.
        tpe_samples: List[Individual] = []
        for ind in self.samples:
            tpe_samples.append(Individual(params=self.param_space.tpe_parameters(ind.get_params()),
                                          performance=copy.deepcopy(ind.get_performance())))
        assert len(tpe_samples) == len(self.samples), "TPE samples size mismatch."

        generations = (self.config.evaluations - self.config.pop_size) // self.num_top_cand
        for gen in range(generations):
            # Log best, average, and median objective values in the current sample set
            self.logger.log_generation(gen, self.config.pop_size + gen * self.num_top_cand,
                                       self.samples, f"{self.surrogate_type}BO")

            # Fit the surrogate to the observed data
            self.surrogate.fit(tpe_samples, self.param_space, self.config.rng)

            # Sample enough candidates from the surrogate to keep the number of 'soft' evaluations consistent between BO and TPEC
            candidates = self.surrogate.sample(self.config.num_candidates * self.num_top_cand, self.param_space, self.config.rng)

            # Select the top candidate(s) for evaluation on the true objective
            best_candidates, _, ei_scores, soft_eval_count = self.surrogate.suggest(self.param_space, candidates, self.num_top_cand)
            if self.config.debug: self.soft_eval_count += soft_eval_count

            # Log per-iteration expected improvement statistics (only from the chosen candidates)
            self.logger.log_ei(gen, self.config.pop_size + gen * self.num_top_cand, ei_scores.tolist())

            # As "candidates" in BO don't have "fixed" counterparts prior to this step.
            # Make sure chosen candidate(s) align with scikit-learn's requirements before evaluation
            for ind in best_candidates: # 'best_candidates' may contain more than 1 Individual
                self.param_space.eval_parameters(ind.get_params()) # fixes in-place

            # Evaluate the best candidates of samples with Ray
            evaluation(best_candidates, X_train_ref, y_train_ref, self.splits, self.config.model, self.config.seed)
            if self.config.debug: self.hard_eval_count += len(best_candidates)

            # remove the best candidate individuals with positive performance
            best_candidates = remove_failed_individuals(best_candidates, self.config)
            # Update sample set
            self.samples += best_candidates
            # Update best performance and set of best performers
            self.best_performance, self.best_performers = process_population_for_best(self.samples, self.best_performance, self.best_performers)
            print(f"Samples size at gen {gen}: {len(self.samples)}", flush=True)
            print(f"Best training performance so far: {self.best_performance}", flush=True)

            # Update modified copy of sample set for fitting surrogate
            tpe_best_candidates: List[Individual] = [copy.deepcopy(ind) for ind in best_candidates]
            for ind in tpe_best_candidates:
                ind.set_params(self.param_space.tpe_parameters(ind.get_params()))
            tpe_samples += tpe_best_candidates

        # randomly select one of the tied best individuals
        assert len(self.best_performers) > 0, "No best performers found in the population."
        best_ind_params = self.config.rng.choice(self.best_performers)

        tpe_best_ind_params = self.param_space.tpe_parameters(best_ind_params)

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
        self.logger.save_tpe_params(self.config, f"{self.surrogate_type}BO", tpe_best_ind_params)

        if self.config.debug:
            print(f"Hard evaluations: {self.hard_eval_count}", flush=True)
            print(f"Soft evaluations {self.soft_eval_count}", flush=True)
            assert(len(self.samples) == self.config.evaluations)

        ray.shutdown()