from abc import ABC, abstractmethod
import numpy as np
from typeguard import typechecked
from typing import Tuple, Dict, List, TypedDict, Any, Literal, Union

# Defining custom type alias
# I believe TypedDict maps the dictionary strings to these keys
# Literal says a value must be exactly one of the specified literals
class IntParam(TypedDict):
    bounds: Tuple[int, int]
    type: Literal["int"] 

class FloatParam(TypedDict):
    bounds: Tuple[float, float]
    type: Literal["float"]

class CatParam(TypedDict):
    bounds: Tuple[str, ...]
    type: Literal["cat"]

class BoolParam(TypedDict):
    bounds: Tuple[bool, bool]
    type: Literal["bool"]

# ParamSpec can be one of...
ParamSpec = Union[IntParam, FloatParam, CatParam, BoolParam]
# Dictionary where each key is a parameter name, and each value is exactly one of the 3 kinds of ParamSpecs
ParamSpace = Dict[str, ParamSpec]


@typechecked
class ModelParams(ABC):
    """
    Genotype for the Organism class.
    
    This class encapsulates a set of hyperparameters and provides
    helper methods for mutation and random sampling
    """
    def __init__(self, random_state: int, param_space: ParamSpace, model_params: Dict[str, Any]):
        self.random_state = random_state  # random_state is a seed
        self.param_space = param_space
        self.model_params = model_params # the genotype

    def __repr__(self):
        # you can simply print the class without accessing self.model_params
        return f"{self.model_params}" 

    def get_parameter_space(self) -> ParamSpace:
        """ Returns the parameter space. """
        return self.param_space

    # function to generate a random set of parameters for the model
    @abstractmethod
    def generate_random_parameters(self) -> Dict[str, Any]:
        """ Returns a random set of parameters. """
        pass

    # function to get the specific type of a parameter within a model
    # must ignore the random_state parameter
    @abstractmethod
    def get_param_type(self, key: str) -> str:
        """ Returns the type of a given parameter name. """
        pass

    @abstractmethod
    def get_params_by_type(self, type: str) -> Dict[str, Any]:
        """ Retrieves a subset of parameters of a given type. """
        pass

    # function to shift float paramters either up or down
    def shift_float_parameter(self, cur_value: float, min: float, max: float, rng_: np.random.default_rng) -> float:
        """ Shifts a float parameter either up or down within bounds. """
        # rng = np.random.default_rng(rng_)
        # 68% of increases/decreases will be within 5% of the current value
        # 95% of increases/decreases will be within 10% of the current value
        # 99.7% of increases/decreases will be within 15% of the
        value = float(cur_value * rng_.normal(1.0, 0.05))

        # ensure the value is within the bounds
        if value < min:
            return min
        elif value > max:
            return max
        else:
            return value

    # function to shift integer parameters either up or down
    def shift_int_parameter(self, cur_value: int, min: int, max: int, rng_: np.random.default_rng) -> int:
        """ Shifts an integer parameter either up or down within bounds. """
        # rng = np.random.default_rng(rng_)
        # 68% of increases/decreases will be within 5% of the current value
        # 95% of increases/decreases will be within 10% of the current value
        # 99.7% of increases/decreases will be within 15% of the
        value = int(cur_value * rng_.normal(1.0, 0.05))

        # ensure the value is within the bounds
        if value < min:
            return min
        elif value > max:
            return max
        else:
            return value

    # function to pick a new value from a categorical parameter
    def pick_categorical_parameter(self, choices: List, rng_: np.random.default_rng):
        """ Picks a random value from a list of categorical choices. """
        # rng = np.random.default_rng(rng_)
        # pick a new value from the choices
        return rng_.choice(choices)

    # function to fix any parameters that do not align with scikit-learn's requirements
    @abstractmethod
    def fix_parameters(self, rng_: np.random.default_rng) -> None:
        """ Fixes parameters that do not align with scikit-learn's requirements. """
        pass

# create a RandomForest subclass that inherits from ModelParams
class RandomForestParams(ModelParams):
    def __init__(self, random_state: int, rng_: np.random.default_rng, params: Dict[str, Any] = {}):
        rng = np.random.default_rng(rng_)
        self.model_params: Dict[str, Any]= {} # the genotype

        self.param_space =  {
            'n_estimators': {'bounds': (10, 1000), 'type': 'int'}, # int
            'criterion': {'bounds': ('gini', 'entropy', 'log_loss'), 'type': 'cat'}, # categorical
            'max_depth': {'bounds': (1, 30), 'type': 'int'}, # int
            'min_samples_split': {'bounds': (.001, 1.0), 'type': 'float'}, # float
            'min_samples_leaf': {'bounds': (.001, 1.0), 'type': 'float'}, # float
            'max_features': {'bounds': (.001, 1.0), 'type': 'float'}, # float
            'max_leaf_nodes': {'bounds': (2, 1000), 'type': 'int'}, # int
            'bootstrap': {'bounds': (True, False), 'type': 'bool'},  # boolean
            'max_samples': {'bounds': (.001, 1.0), 'type': 'float'},  # float
            'random_state': random_state,  # int
        }
        super().__init__(random_state=random_state, param_space=self.param_space, model_params=self.model_params)

        # if params is empty, get a random set of parameters to initialize the genotype
        if len(params) == 0:
            self.model_params = self.generate_random_parameters(rng)
            # fix the parameters to ensure they are valid
            self.fix_parameters(rng)
        else:
            # otherwise, use the provided parameters
            self.model_params = params
    
    def mutate_parameters(self, rng_: np.random.default_rng, mut_rate: float = 0.1) -> None:
        """
        Mutates the model parameters (genotype) in-place with a given mutation rate.

        Parameters: 
            rng_ (np.random.default_rng): A NumPy random generator instance.
            mut_rate (float): Probability of mutating each parameter.
        """
        # Per-gene mutation
        for name, spec in self.param_space.items():
            # Coin flip to decide whether to mutate each parameter
            if rng_.uniform() < mut_rate:
                if spec["type"] == "int":
                    self.model_params[name] = self.shift_int_parameter(self.model_params[name], spec['bounds'][0], spec['bounds'][1], rng_)
                elif spec["type"] == "float":
                    self.model_params[name] = self.shift_float_parameter(self.model_params[name], spec['bounds'][0], spec['bounds'][1], rng_)
                elif spec["type"] in ["cat", "bool"]:
                    self.model_params[name] = self.pick_categorical_parameter(spec['bounds'], rng_)

        # Fix the parameters to ensure they are valid
        self.fix_parameters(rng_)
    
    
    def generate_random_parameters(self, rng_: np.random.default_rng) -> Dict[str, Any]:
        """
        Generate a random set of parameters based on the defined parameter space.

        Parameters:
            rng_ (np.random.default_rng): A NumPy random generator instance.
        Returns:
            Dict[str, Any]: A dictionary of randomly generated parameters.
        """
        rand_genotype = {}
        for param_name, spec in self.param_space.items():
            if spec["type"] == "int":
                rand_genotype[param_name] = rng_.integers(*spec["bounds"])
            elif spec["type"] == "float":
                rand_genotype[param_name] = float(rng_.uniform(*spec["bounds"]))
            elif spec["type"] in {"cat", "bool"}:
                rand_genotype[param_name] = rng_.choice(spec["bounds"])
            else:
                raise ValueError(f"Unsupported parameter type: {spec['type']}")
        rand_genotype["random_state"] = self.param_space['random_state']
        return rand_genotype
    
    def get_param_type(self, key: str) -> str:
        """ Returns the type of a given parameter. """
        # This should automatically raise a KeyError if 'key' does not exist
        return self.param_space[key]['type']
    
    def get_params_by_type(self, type: str) -> Dict[str, Any]:
        """ Retrieves a subset of parameters of a given type. """
        if type not in ['int', 'float', 'cat', 'bool']:
            raise ValueError(f"Unsupported parameter type: {type}")
        return {name: info for name, info in self.param_space.items() 
                if info['type'] == type}

    def fix_parameters(self, rng_: np.random.default_rng) -> None:
        # if bootstrap is False, we need to set max_samples to None
        if not self.model_params['bootstrap']:
            self.model_params['max_samples'] = None
        return