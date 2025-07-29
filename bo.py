import numpy as np
from config import Config
from logger import Logger
from param_space import ModelParams
from organism import Organism
from surrogate import Surrogate
from tpe import TPE
from typeguard import typechecked
from typing import Tuple, Dict, List, Literal

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

        self.samples: List[Organism] = [] # This stores the observed samples

        # For debugging
        self.hard_eval_count = 0
        self.soft_eval_count = 0
    
    def get_samples(self) -> List[Organism]:
        return self.samples
    
    def set_samples(self, samples: List[Organism]):
        self.samples = samples

    def run(self, X_train, y_train):
        # Initialize observed sample set, its size is determined by the 'pop_size' parameter in the Config
        self.samples = [Organism(self.param_space) for _ in range(self.config.pop_size)]

        # Evaluate initial samples on the true objective function (cross-validation)
        for org in self.samples:
            # Invert to minimize
            org.set_fitness(-1 * self.param_space.eval_parameters(org.get_genotype(), X_train, y_train))
            if self.config.debug: self.hard_eval_count += 1

        generations = (self.config.evaluations - self.config.pop_size) // self.num_top_cand
        for gen in range(generations):
            # Log best, average, and median objective values in the current sample set
            self.logger.log_generation(gen, self.config.pop_size + gen * self.num_top_cand, 
                                       self.samples, f"{self.surrogate_type}BO")

            # Fit the surrogate to the observed data
            self.surrogate.fit(self.samples, self.param_space)

            # We randomly select enough candidates to keep the number of 'soft' evaluations consistent between BO and TPEC
            # We define 'soft' evaluations to be those performed on the surrogate
            # candidates = [Organism(self.param_space) for _ in range(self.config.num_candidates * self.num_top_cand)]
            candidates = self.surrogate.sample(self.config.num_candidates * self.num_top_cand, self.param_space)

            # Select the top candidate(s) for evaluation on the true objective
            best_org, ei_scores, soft_eval_count = self.surrogate.suggest(self.param_space, candidates, self.num_top_cand)
            assert(len(best_org) == self.num_top_cand)
            if self.config.debug: self.soft_eval_count += soft_eval_count

            # Log per-iteration expected improvement statistics (only from the chosen candidates)
            self.logger.log_ei(gen, self.config.pop_size + gen * self.num_top_cand, ei_scores)

            # Evaluate the chosen candidates on the true objective
            for org in best_org: # 'best_org' may contain more than 1 organism
                org.set_fitness(-1 * self.param_space.eval_parameters(org.get_genotype(), X_train, y_train))
                if self.config.debug: self.hard_eval_count += 1 

            # Update sample set
            self.samples += best_org

        # For the final generation
        self.samples.sort(key=lambda o: o.get_fitness())
        best_org = self.samples[0]
        self.param_space.fix_parameters(best_org.get_genotype())

        # Log best, average, and median objective values in the final sample set
        self.logger.log_generation(generations, self.config.pop_size + generations * self.num_top_cand, 
                                   self.samples, f"{self.surrogate_type}BO")
        # Log the best observed hyperparameter configuration across all iterations
        self.logger.log_best(best_org, self.config, f"{self.surrogate_type}BO")
        self.logger.save(self.config, f"{self.surrogate_type}BO")

        if self.config.debug:
            print(f"Hard evaluations: {self.hard_eval_count}")
            print(f"Soft evaluations {self.soft_eval_count}")
            assert(len(self.samples) == self.config.evaluations)