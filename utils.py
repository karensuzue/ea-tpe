import os
import pickle
import copy
import ray
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Tuple, Optional
from param_space import RandomForestParams, ModelParams

def eval_parameters_RF_final(model_params: Dict[str, Any], X_train, y_train, X_test, y_test, seed: int) -> Tuple[float, float]:
    model = RandomForestClassifier(**model_params, random_state = seed)
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    return train_accuracy, test_accuracy

def eval_parameters_RF(model_params: Dict[str, Any], X_train, y_train, seed: int, index: int) -> Tuple[float, int]:
    """
    Evaluates a given set of hyperparameters on cross-validated accuracy.

    Parameters:
        model_params (Dict[str, Any]): The set of hyperparameters to evaluate.
    """
    # Must use the same seed/random_state across parameters and methods,
    # to maintain the same CV splits
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) # initialize our cv splitter
    # Both model internals and data splits are reproducible
    model = RandomForestClassifier(**model_params, random_state=seed)

    try:
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    except Exception as e:
        print(f"Error evaluating model {index} parameters {model_params}: {e}")
        return 1.0, index  # Return a high score to avoid selecting this individual
    return -1.0 * score, index  # minimize

def eval_final_factory(model: str, model_params: Dict[str, Any], X_train, y_train, X_test, y_test, seed: int) -> Tuple[float, float]:
    if model == 'RF':
        # print("Final evaluations for RF") # debug
        return eval_parameters_RF_final(model_params, X_train, y_train, X_test, y_test, seed)
    else:
        raise ValueError(f"Unsupported model: {model}")

def bo_eval_parameters_RF(model_params: Dict[str, Any], X_train, y_train, seed: int, n_jobs: int) -> float:
    """
    Evaluates a given set of hyperparameters on cross-validated accuracy.

    Parameters:
        model_params (Dict[str, Any]): The set of hyperparameters to evaluate.
    """
    # Must use the same seed/random_state across parameters and methods,
    # to maintain the same CV splits
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) # initialize our cv splitter
    # Both model internals and data splits are reproducible
    model = RandomForestClassifier(**model_params, random_state=seed, n_jobs=n_jobs)

    try:
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    except Exception as e:
        print(f"Error evaluating model parameters {model_params}: {e}")
        return 1.0  # Return a high score to avoid selecting this individual
    return -1.0 * score  # minimize

@ray.remote
def eval_factory(model: str, model_params: Dict[str, Any], X_train, y_train,
                 seed: int, index: int) -> Tuple[float, int]:
    """
    Computes the performance score for the given set of hyperparameters under the specified model type.

    Parameters:
        model (str): The name of the model to evaluate (e.g., 'RF').
        model_params (Dict[str, Any]): The hyperparameters for the model.
        X_train, y_train: The training data.
        seed (int): Random seed for reproducibility.
        index (Optional[int]): Index of the individual in the population (used for Ray-based parallel evaluation).

    Returns:
        Tuple[float, Optional[int]]: The performance score and (optionally) the individual's index.
    """
    if model == 'RF':
        # print("Evaluation for RF.") # debug
        return eval_parameters_RF(model_params, X_train, y_train, seed, index)
    else:
        raise ValueError(f"Unsupported model: {model}")

def param_space_factory(model: str, rng: np.random.default_rng) -> ModelParams:
    if model == 'RF':
        # print("RF Parameter space chosen.") # debug
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
