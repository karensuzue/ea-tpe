import numpy as np
import openml
import random
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, List, TypedDict, Literal, Union


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
                 tour_size: int = 5,
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
        # self.replacement = replacement
        self.dataset_idx = dataset_idx
        self.logdir = logdir
        self.debug = debug

        self.dataset_ids = [1464, 1489, 44]
        self.param_names = ['n_estimators', 'criterion']
        self.param_space = {
            'n_estimators': {'bounds': (50, 500), 'type': 'int'},
            'criterion': {'bounds': ("gini", "entropy", "log_loss"), 'type': 'cat'}
        }

        # This should affect the entire system
        random.seed(seed)
        np.random.seed(seed)
    
    def get_seed(self) -> int:
        return self.seed
    
    def get_evaluations(self) -> int:
        """ Get the number of evaluations. """
        return self.evaluations
    
    def get_pop_size(self) -> int:
        """ Get population size. """
        return self.pop_size
    
    def get_num_candidates(self) -> int:
        """ For TPE and EA+TPE. """
        return self.num_candidates
    
    def get_tour_size(self) -> int:
        """ Get tournament size for selection. """
        return self.tour_size
    
    def get_mut_rate(self) -> float:
        """ Get mutation rate. """
        return self.mut_rate
    
    def get_param_space(self) -> ParamSpace:
        """ Get parameter names and specifications. """
        return self.param_space
    
    def get_param_names(self) -> List[str]:
        """ Get parameter names. """
        return self.param_names
    
    def get_num_param_names(self) -> List[str]:
        return [n for n, info in self.param_space.items() 
                if info['type'] == "int" or info['type'] == "float"]
    
    def get_cat_param_names(self) -> List[str]:
        return [n for n, info in self.param_space.items()
                if info['type'] == "cat"]
    
    def get_param_type(self, param_name) -> str:
        # This should automatically raise a KeyError if 'param_name' does not exist
        return self.param_space[param_name]['type']
    
    def add_param(self, name: str, bounds: Tuple[Any, Any], type: Literal["int", "float", "cat"]) -> None:
        """ Adds a new parameter. """
        self.param_space[name] = {'bounds': bounds, 'type': type}
        self.param_names.append(name)
    
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
        return self.logdir
    
    def get_debug(self) -> bool:
        return self.debug