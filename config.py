import numpy as np
import openml
import random
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, List, TypedDict, Literal, Union
from logger import Logger


# Defining custom type alias
# I believe TypedDict maps the dictionary strings to these keys
class IntParam(TypedDict):
    bounds: Tuple[int, int]
    type: Literal["int"]

class FloatParam(TypedDict):
    bounds: Tuple[float, float]
    type: Literal["float"]

class CatParam(TypedDict):
    bounds: Tuple[str, ...]
    type: Literal["cat"]

# Missing: Boolean, will handle later

# ParamSpec can be one of...
ParamSpec = Union[IntParam, FloatParam, CatParam]
# Dictionary where each key is a parameter name, and each value is exactly one of the 3 kinds of ParamSpecs
ParamSpace = Dict[str, ParamSpec]

class Config:
    def __init__(self, 
                 seed: int = 0,
                 evaluations: int = 500,
                 pop_size: int = 50, 
                 num_candidates: int = 10,
                 tour_size: int = 2,
                 mut_rate: float = 0.1,
                 # replacement: bool = True,
                 dataset_idx: int = 0, 
                 logdir: str = 'results',
                 debug = False):
        
        # if num_child < pop_size:
        #     raise ValueError(f"'num_child' ({num_child}) must be >= 'pop_size' ({pop_size})")
        
        self.seed = seed
        self.evaluations = evaluations
        self.pop_size = pop_size
        self.num_candidates = num_candidates
        self.tour_size = tour_size
        self.mut_rate = mut_rate
        self.dataset_idx = dataset_idx
        self.logdir = logdir
        self.debug = debug

        self.dataset_ids = [1464, 1489, 44]
        # self.param_names = ['n_estimators', 'criterion'] # Not necessary
        self.param_space = { # use ConfigSpace in the future
            'n_estimators': {'bounds': (50, 500), 'type': 'int'},
            'criterion': {'bounds': ("gini", "entropy", "log_loss"), 'type': 'cat'}
            # ... add more if needed
        }

        self.logger = Logger(self.logdir)

        # This should affect the entire system
        random.seed(seed)
        np.random.seed(seed)
    
    def get_seed(self) -> int:
        """ Returns the current seed. """
        return self.seed
    
    def get_evaluations(self) -> int:
        """ Returns the number of evaluations. """
        return self.evaluations
    
    def get_pop_size(self) -> int:
        """ Returns population size. """
        return self.pop_size
    
    def get_num_candidates(self) -> int:
        """ Returns the number of candidate "offspring" For TPE and EA+TPE. """
        return self.num_candidates
    
    def get_tour_size(self) -> int:
        """ Returns tournament size for selection. """
        return self.tour_size
    
    def get_mut_rate(self) -> float:
        """ Returns mutation rate. """
        return self.mut_rate
    
    
    def get_dataset_id(self) -> int:
        """ Returns the ID of the chosen dataset. """
        return self.dataset_idx
    
    def get_dataset_name(self) -> int:
        """ Returns the OpenML ID of the chosen dataset. """
        return self.dataset_ids[self.dataset_idx]

    def add_dataset_id(self, id: int) -> None:
        """ Adds a new dataset ID to the list of available datasets. """
        self.dataset_ids.append(id)

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        """ Loads and splits the chosen dataset. """
        if self.dataset_idx >= len(self.dataset_ids):
            raise ValueError(f"Dataset index {self.dataset_idx} is invalid.")
        
        dataset = openml.datasets.get_dataset(self.dataset_ids[self.dataset_idx])
        df, *_ = dataset.get_data()

        # Minor correction for dataset 44
        if self.dataset_idx == 2:
            df.rename(columns = {'class': 'Class'}, inplace = True)

        X = df.drop(columns = 'Class').values
        y = df['Class'].values

        # Make sure dataset splits use the same seed across all experiments/replicates
        return train_test_split(X, y, test_size = 0.2, random_state = 0)

    def get_logdir(self) -> str:
        """ Returns the log directory path. """
        return self.logdir
    
    def get_logger(self) -> Logger:
        """ Returns the Logger object. """
        return self.logger
    
    def get_debug(self) -> bool:
        """ Returns 'debug' state. """
        return self.debug