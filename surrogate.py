import numpy as np
from organism import Organism
from abc import ABC, abstractmethod
from typeguard import typechecked
from typing import List, Tuple

@typechecked
class Surrogate(ABC):
    """
    Pluggable surrogate model for the Bayesian Optimizer.
    """
    @abstractmethod
    def fit(self, samples: List[Organism]):
        """ 
        Fit the model to a set of observations.

        Parameters:
        """
        pass

    @abstractmethod
    def suggest(self, candidates: List[Organism], num_top_cand: int) -> Tuple[List[Organism], np.ndarray, int]:
        """
        Suggest the top candidate(s) after ranking them according to an acquisition score.

        Parameters:
            candidates (List[Organism]): Candidate hyperparameter sets to rank.
            num_top_cand (int): Number of top candidates to return. 

        Returns:
            Tuple[List[ModelParams], np.ndarray, int]: A tuple containing the top-k candidates, their scores, 
                                                        and the number of soft evaluations performed.
        """
        pass