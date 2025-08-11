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

# ParamSpec can be one of IntParam, FloatParam, CatParam, and BoolParam
ParamSpec = Union[IntParam, FloatParam, CatParam, BoolParam]
# Dictionary where each key is a parameter name, and each value is exactly one of the 3 kinds of ParamSpecs
ParamSpace = Dict[str, ParamSpec] # {parameter_name: {"bounds": Tuple, "type": Literal["int", "float", "cat", "bool"]}}

@typechecked
class ModelParams(ABC):
    """
    This class encapsulates the parameter space and provides
    helper methods for mutation and random sampling.
    """
    def __init__(self, param_space: ParamSpace, rng: np.random.default_rng):
        # self.random_state = random_state  # random_state is a seed
        self.param_space = param_space
        self.rng = rng # already seeded

    def get_parameter_space(self) -> ParamSpace:
        """ Returns the parameter space. """
        return self.param_space

    # function to generate a random set of parameters
    # Format {parameter_name: value}
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
    def get_params_by_type(self, type: str) -> Dict:
        """ Retrieves a subset of parameters of a given type. """
        pass

    # function to shift float paramters either up or down
    def shift_float_parameter(self, cur_value: float, min: float, max: float) -> float:
        """ Shifts a float parameter either up or down within bounds. """
        # rng = np.random.default_rng(rng_)
        # 68% of increases/decreases will be within 5% of the current value
        # 95% of increases/decreases will be within 10% of the current value
        # 99.7% of increases/decreases will be within 15% of the
        value = float(cur_value * self.rng.normal(1.0, 0.05))

        # ensure the value is within the bounds, clip to safe boundaries
        eps = 1e-12
        return np.clip(value, min + eps, max - eps)

    # function to shift integer parameters either up or down
    def shift_int_parameter(self, cur_value: int, min: int, max: int) -> int:
        """ Shifts an integer parameter either up or down within bounds. """
        # rng = np.random.default_rng(rng_)
        # 68% of increases/decreases will be within 5% of the current value
        # 95% of increases/decreases will be within 10% of the current value
        # 99.7% of increases/decreases will be within 15% of the
        value = int(cur_value * self.rng.normal(1.0, 0.05))

        # ensure the value is within the bounds
        if value < min:
            return min
        elif value > max:
            return max
        else:
            return value

    # function to pick a new value from a categorical parameter
    def pick_categorical_parameter(self, choices: List | Tuple):
        """ Picks a random value from a list of categorical choices. """
        # rng = np.random.default_rng(rng_)
        # pick a new value from the choices
        return self.rng.choice(choices)

    @abstractmethod
    def mutate_parameters(self, model_params: Dict[str, Any], mut_rate: float = 0.1) -> None:
        """ Mutates a given set of hyperparameters in-place. """
        pass

    # function to fix any parameters that do not align with scikit-learn's requirements
    @abstractmethod
    def fix_parameters(self, model_params: Dict[str, Any]) -> None:
        """ Fixes parameters (in-place) that do not align with scikit-learn's requirements. """
        pass

    @abstractmethod
    def eval_parameters(self, model_params: Dict[str, Any], X_train, y_train) -> float:
        """ Evaluates a given set of hyperparameters against some metric. """
        pass

# create a RandomForest subclass that inherits from ModelParams
class RandomForestParams(ModelParams):
    def __init__(self, rng: np.random.default_rng, offset: float = 1e-6):
        self.rng = rng

        self.param_space =  {
            'n_estimators': {'bounds': (10, 1000), 'type': 'int'}, # int
            'criterion': {'bounds': ('gini', 'entropy', 'log_loss'), 'type': 'cat'}, # categorical
            'max_depth': {'bounds': (1, 30), 'type': 'int'}, # int
            'min_samples_split': {'bounds': (.001, 1.0 - offset), 'type': 'float'}, # float
            'min_samples_leaf': {'bounds': (.001, 1.0 - offset), 'type': 'float'}, # float
            'max_features': {'bounds': (.001, 1.0 - offset), 'type': 'float'}, # float
            'max_leaf_nodes': {'bounds': (2, 1000), 'type': 'int'}, # int
            'bootstrap': {'bounds': (True, False), 'type': 'bool'},  # boolean
            'max_samples': {'bounds': (.001, 1.0 - offset), 'type': 'float'},  # float

            # a copy of 'max_samples' that is unaffected/not used by surrogate model
            'max_samples_og': {'bounds': (.001, 1.0 - offset), 'type': 'special-float'} 
        }
        super().__init__(param_space=self.param_space, rng=self.rng)

    def mutate_parameters(self, model_params: Dict[str, Any], mut_rate: float = 0.1) -> None:
        """
        Mutates the model parameters (genotype) in-place with a given mutation rate.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to mutate.
            mut_rate (float): Probability of mutating each parameter.
        """
        # Per-gene mutation
        for name, spec in self.param_space.items():
            # Coin flip to decide whether to mutate each parameter
            if self.rng.uniform() < mut_rate:
                if spec["type"] == "int":
                    model_params[name] = self.shift_int_parameter(int(model_params[name]), spec['bounds'][0], spec['bounds'][1])
                elif spec["type"] == "float":
                    if name == "max_samples":
                        # if model_params['bootstrap'] is False: model_params[name] = None
                        # elif model_params['bootstrap'] is True and model_params['max_samples_og'] is None:
                        #     model_params[name] = new random num
                        # elif model_params['bootstrap'] is True and model_params['max_samples_og'] is not None:
                        #     assert(model_params['max_samples_og'] == model_params[name])
                        #     self.shift_float_parameter()
                    else:
                        model_params[name] = self.shift_float_parameter(float(model_params[name]), spec['bounds'][0], spec['bounds'][1])
                elif spec["type"] in ["cat", "bool"]:
                    model_params[name] = self.pick_categorical_parameter(spec['bounds'])

        # Fix the parameters to ensure they are valid
        self.fix_parameters(model_params)
        return

    def generate_random_parameters(self) -> Dict[str, Any]:
        """
        Generates a random set of parameter values based on the defined parameter space.

        Parameters:
            rng_ (np.random.default_rng): A NumPy random generator instance.
        Returns:
            Dict[str, Any]: A dictionary of randomly generated parameters.
        """
        rand_genotype = {}
        for param_name, spec in self.param_space.items():
            if spec["type"] == "int":
                rand_genotype[param_name] = int(self.rng.integers(*spec["bounds"]))
            elif spec["type"] == "float":
                rand_genotype[param_name] = float(self.rng.uniform(*spec["bounds"]))
            elif spec["type"] in {"cat", "bool"}:
                rand_genotype[param_name] = self.rng.choice(spec["bounds"])
            elif param_name == "max_samples_og": # don't generate new values for this
                rand_genotype[param_name] = rand_genotype["max_samples"]
            else:
                raise ValueError(f"Unsupported parameter type: {spec['type']}")
        # Fix the parameters to ensure they are valid
        self.fix_parameters(rand_genotype)
        return rand_genotype

    def get_param_type(self, key: str) -> str:
        """ Returns the type of a given parameter. """
        # This should automatically raise a KeyError if 'key' does not exist
        return self.param_space[key]['type']

    def get_params_by_type(self, type: str) -> Dict:
        """ Retrieves a subset of parameters of a given type. """
        if type not in ['int', 'float', 'cat', 'bool', 'special-float']:
            raise ValueError(f"Unsupported parameter type: {type}")
        return {name: info for name, info in self.param_space.items() if info['type'] == type}

    # @staticmethod
    def fix_parameters(self, model_params: Dict[str, Any]) -> None:
        """ Fixes parameters (in-place) that do not align with scikit-learn's requirements. """
        # if bootstrap is False, we need to set max_samples to None
        if not model_params['bootstrap']:
            model_params['max_samples'] = None
            model_params['max_samples_og'] = None
        if model_params['bootstrap'] and model_params['max_samples'] is None:
            model_params['max_samples'] = float(self.rng.uniform(*self.param_space['max_samples']['bounds']))
            model_params['max_samples_og'] = model_params['max_samples']
        return
    
    def tpe_parameters(self) -> Dict[str, Any]:
        # return deepcopy of parameters with modification for TPE fits
        # assert if bootstrap is true, max_samples can't be none
            # additional bounds checking?
        # assert if bootstrap is false, max-sample has to be none 


        