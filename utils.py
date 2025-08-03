import os
import pickle
import copy
import ray
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Tuple
from param_space import RandomForestParams, ModelParams

def eval_parameters_RF_final(model_params: Dict[str, Any], X_train, y_train, X_test, y_test) -> Tuple[float, float]:
    model_params_copy = copy.deepcopy(model_params)
    RandomForestParams.fix_parameters(model_params_copy)
    model = RandomForestClassifier(**model_params_copy, random_state=0)
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    return train_accuracy, test_accuracy

@ray.remote
def eval_parameters_RF(model_params: Dict[str, Any], X_train, y_train, seed: int) -> float:
    """ 
    Evaluates a given set of hyperparameters on cross-validated accuracy.
    
    Parameters:
        model_params (Dict[str, Any]): The set of hyperparameters to evaluate.
    """
    # "fix_parameters()" changes "model_params" in-place, so a copy must be made
    # We also must retain the original "model_params" for TPE's fit()
    model_params_copy = copy.deepcopy(model_params)
    RandomForestParams.fix_parameters(model_params_copy)

    # Must use the same seed/random_state across parameters and methods, 
    # to maintain the same CV splits 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) # initialize our cv splitter
    # Both model internals and data splits are reproducible
    model = RandomForestClassifier(**model_params_copy, random_state=seed)
    
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean() 
    return -1 * score # minimize


def eval_final_factory(model: str, model_params: Dict[str, Any], X_train, y_train, X_test, y_test) -> Tuple[float, float]:
    if model == 'RF':
        return eval_parameters_RF_final(model_params, X_train, y_train, X_test, y_test)
    else:
        raise ValueError(f"Unsupported model: {model}")
    

def eval_factory(model: str, model_params: Dict[str, Any], X_train, y_train) -> float:
    if model == 'RF':
        return eval_parameters_RF(model_params, X_train, y_train)
    else:
        raise ValueError(f"Unsupported model: {model}")

def param_space_factory(model: str, rng: np.random.default_rng) -> ModelParams:
    if model == 'RF':
        return RandomForestParams(rng)
    # elif model == 'XGBoost': # TENTATIVE
    #     return XGBoostParams(seed)
    else:
        raise ValueError(f"Unsupported model: {model}")
    
#https://github.com/automl/ASKL2.0_experiments/blob/84a9c0b3af8f7ac6e2a003d4dea5e6dce97d4315/experiment_scripts/utils.py
def load_task(task_id: int, data_dir: str, preprocess=True):
    """ 
    Loads and splits the chosen task. 
    Project must include 'data' directory, which stores a set of 
    preprocessed and cached OpenML tasks. 
    """
    cached_data_path = f"{data_dir}/{task_id}_{preprocess}.pkl"
    if os.path.exists(cached_data_path):
        d = pickle.load(open(cached_data_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']
    else:
        print(f'Task {task_id} not found')
        exit(0)

    return X_train, y_train, X_test, y_test
