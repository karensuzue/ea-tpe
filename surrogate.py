import numpy as np
from individual import Individual
from param_space import ModelParams
from abc import ABC, abstractmethod
from typeguard import typechecked
from typing import List, Tuple

@typechecked
class Surrogate(ABC):
    """
    Pluggable surrogate model for the Bayesian Optimizer.
    """
    @abstractmethod
    def fit(self, samples: List[Individual], param_space: ModelParams, rng: np.random.default_rng) -> None:
        """
        Fit the model to a set of observations.

        Parameters:
        """
        pass

    @abstractmethod
    def suggest(self, param_space: ModelParams, candidates: List[Individual], num_top_cand: int) -> Tuple[List[Individual], np.ndarray, np.ndarray, int]:
        """
        Suggest the top candidate(s) after ranking them according to an acquisition score.

        Parameters:
            candidates (List[Individual]): Candidate hyperparameter sets to rank.
            num_top_cand (int): Number of top candidates to return.

        Returns:
            Tuple[List[ModelParams], np.ndarray, int]: A tuple containing the top-k candidates, their scores,
                                                        and the number of soft evaluations performed.
        """
        pass

    @abstractmethod
    def sample(self, num_samples: int, param_space: ModelParams, rng: np.random.default_rng) -> List[Individual]:
        """
        Returns 'num_samples' samples
        param_space may be necessary to figure out which parameter a value belongs to.
        """