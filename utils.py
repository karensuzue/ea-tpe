import os
import pickle
import copy
import ray
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from typing import Dict, Any, Tuple, List
from typeguard import typechecked
from param_space import LinearSGDParams, GradientBoostParams, ExtraTreesParams, KernelSVCParams, DecisionTreeParams, LinearSVCParams, RandomForestParams, ModelParams
from individual import Individual
from config import Config

###########################################################
#                        FACTORIES                        #
###########################################################
# function to take in a population, X and y train ray ids, and cross-validation splits to parallel evaluate the population
@typechecked
def evaluation(population: List[Individual], 
               X_train: ray.ObjectID, 
               y_train: ray.ObjectID, 
               cv_splits: List, 
               model: str, 
               seed: int) -> None:
    # create a dictionary to hold each individual's in the population scores across each fold
    individual_scores = {i: {'cv': [], 'error': False} for i in range(len(population))}

    # create a ray job for each individual in the population for each fold in the cross-validation splits
    ray_jobs = []
    for i, individual in enumerate(population):
        for _, (train_split, validation_split) in enumerate(cv_splits):
            if model == 'RF':
                ray_jobs.append(ray_RF_eval.remote(individual.get_params(), X_train, y_train, train_split, validation_split, seed, i))
            elif model == 'LSVC':
                ray_jobs.append(ray_LSVC_eval.remote(individual.get_params(), X_train, y_train, train_split, validation_split, seed, i))
            elif model == 'DT':
                ray_jobs.append(ray_DT_eval.remote(individual.get_params(), X_train, y_train, train_split, validation_split, seed, i))
            elif model == 'KSVC':
                ray_jobs.append(ray_KSVC_eval.remote(individual.get_params(), X_train, y_train, train_split, validation_split, seed, i))
            elif model == 'ET':
                ray_jobs.append(ray_ET_eval.remote(individual.get_params(), X_train, y_train, train_split, validation_split, seed, i))
            elif model == 'GB':
                ray_jobs.append(ray_GB_eval.remote(individual.get_params(), X_train, y_train, train_split, validation_split, seed, i))
            elif model == 'LSGD':
                ray_jobs.append(ray_LSGD_eval.remote(individual.get_params(), X_train, y_train, train_split, validation_split, seed, i))
            else:
                raise ValueError(f"Unsupported model: {model}")

    # process the ray jobs as they finish
    while len(ray_jobs) > 0:
        done, ray_jobs = ray.wait(ray_jobs, num_returns = 1)
        score, id, error, error_msg = ray.get(done)[0]
        if error < 0.0:
            individual_scores[id]['error'] = True
            print(error_msg)
        individual_scores[id]['cv'].append(score)
    # make sure all individuals have the correct number of folds
    assert all(len(individual_scores[i]['cv']) == len(cv_splits) for i in range(len(population))), "Not all individuals have the correct number of CV folds."

    # set each individual's performance to the mean of their cross-validation scores, or to a positive value if they had an error
    for i, individual in enumerate(population):
        if individual_scores[i]['error']:
            individual.set_performance(1.0) # set to a positive value to indicate failure
        else:
            individual.set_performance(np.mean(individual_scores[i]['cv']))
    return


@typechecked
def eval_final_factory(model: str, 
                       model_params: Dict[str, Any], 
                       X_train: np.ndarray, 
                       y_train: np.ndarray, 
                       X_test: np.ndarray, 
                       y_test: np.ndarray, 
                       seed: int) -> Tuple[float, float]:
    if model == 'RF':
        # print("Final evaluations for RF") # debug
        return eval_parameters_RF_final(model_params, X_train, y_train, X_test, y_test, seed)
    elif model == 'LSVC':
        return eval_parameters_LSVC_final(model_params, X_train, y_train, X_test, y_test, seed)
    elif model == 'DT':
        return eval_parameters_DT_final(model_params, X_train, y_train, X_test, y_test, seed)
    elif model == 'KSVC':
        return eval_parameters_KSVC_final(model_params, X_train, y_train, X_test, y_test, seed)
    elif model == 'ET':
        return eval_parameters_ET_final(model_params, X_train, y_train, X_test, y_test, seed)
    elif model == 'GB':
        return eval_parameters_GB_final(model_params, X_train, y_train, X_test, y_test, seed)
    elif model == 'LSGD':
        return eval_parameters_LSGD_final(model_params, X_train, y_train, X_test, y_test, seed)
    else:
        raise ValueError(f"Unsupported model: {model}")

@typechecked
def param_space_factory(model: str, rng: np.random.default_rng, num_cpus: int, classes: int) -> ModelParams:
    if model == 'RF':
        # print("RF Parameter space chosen.") # debug
        return RandomForestParams(rng)
    elif model == 'LSVC': 
        return LinearSVCParams(rng)
    elif model == 'DT':
        return DecisionTreeParams(rng)
    elif model == 'KSVC':
        return KernelSVCParams(rng)
    elif model == 'ET':
        return ExtraTreesParams(rng)
    elif model == 'GB':
        return GradientBoostParams(rng, classes=classes)
    elif model == 'LSGD':
        return LinearSGDParams(rng, num_cpus)
    else:
        raise ValueError(f"Unsupported model: {model}")
    
# @typechecked
# def eval_factory(model: str, 
#                  model_params: Dict[str, Any], 
#                  X_train: np.ndarray, 
#                  y_train: np.ndarray,
#                  seed: int, n_jobs: int) -> float:
#     if model == 'RF':
#         return eval_parameters_RF(model_params, X_train, y_train, seed, n_jobs)
#     elif model == 'LSVC':
#         return eval_parameters_LSVC(model_params, X_train, y_train, seed, n_jobs)
#     else:
#         raise ValueError(f"Unsupported model: {model}")



###########################################################
#                      RANDOM FOREST                      #
###########################################################
@typechecked
def eval_parameters_RF_final(model_params: Dict[str, Any], 
                             X_train: np.ndarray, 
                             y_train: np.ndarray, 
                             X_test: np.ndarray, 
                             y_test: np.ndarray, 
                             seed: int) -> Tuple[float, float]:
    model = RandomForestClassifier(**model_params, random_state = seed)
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    return train_accuracy, test_accuracy

@ray.remote
def ray_RF_eval(model_params: Dict[str, Any], 
                X_train: np.ndarray, 
                y_train: np.ndarray, 
                train_split,
                validation_split, 
                seed: int, id: int) -> Tuple[float, int, float]:
    # initialize the RF model
    model = RandomForestClassifier(**model_params, random_state=seed)

    # try to fit
    try:
        # fit model on training data split
        model.fit(X_train[train_split], y_train[train_split])
        accuracy = model.score(X_train[validation_split], y_train[validation_split])
        return -1.0 * float(accuracy), id, 1.0, None

    except Exception as e:
        # failed
        return -1.0, id, -1.0, str(e)

# # DEPRECATED
# @typechecked
# def eval_parameters_RF(model_params: Dict[str, Any], 
#                        X_train: np.ndarray, 
#                        y_train: np.ndarray, 
#                        seed: int, n_jobs: int) -> float:
#     """
#     Evaluates a given set of hyperparameters on cross-validated accuracy.

#     Parameters:
#         model_params (Dict[str, Any]): The set of hyperparameters to evaluate.
#     """
#     # Must use the same seed/random_state across parameters and methods,
#     # to maintain the same CV splits
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) # initialize our cv splitter
#     # Both model internals and data splits are reproducible
#     model = RandomForestClassifier(**model_params, random_state=seed, n_jobs=n_jobs)
#     try:
#         score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
#     except Exception as e:
#         print(f"Error evaluating model parameters {model_params}: {e}")
#         return 1.0  # Return a high score to avoid selecting this individual
#     return -1.0 * score  # minimizes
  

###########################################################
#                       LINEAR SVC                        #
###########################################################
@typechecked
def eval_parameters_LSVC_final(model_params: Dict[str, Any], 
                             X_train: np.ndarray, 
                             y_train: np.ndarray, 
                             X_test: np.ndarray, 
                             y_test: np.ndarray, 
                             seed: int) -> Tuple[float, float]:
    model = LinearSVC(**model_params, random_state=seed)
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    return train_accuracy, test_accuracy

@ray.remote
def ray_LSVC_eval(model_params: Dict[str, Any], 
                X_train: np.ndarray, 
                y_train: np.ndarray, 
                train_split,
                validation_split, 
                seed: int, id: int) -> Tuple[float, int, float]:
    
    # initialize the RF model
    model = LinearSVC(**model_params, random_state=seed)

    # try to fit
    try:
        # fit model on training data split
        model.fit(X_train[train_split], y_train[train_split])
        accuracy = model.score(X_train[validation_split], y_train[validation_split])
        return -1.0 * float(accuracy), id, 1.0, None

    except Exception as e:
        # failed
        return -1.0, id, -1.0, str(e)


###########################################################
#                      DECISION TREE                      #
###########################################################
@typechecked
def eval_parameters_DT_final(model_params: Dict[str, Any], 
                             X_train: np.ndarray, 
                             y_train: np.ndarray, 
                             X_test: np.ndarray, 
                             y_test: np.ndarray, 
                             seed: int) -> Tuple[float, float]:
    model = DecisionTreeClassifier(**model_params, random_state=seed)
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    return train_accuracy, test_accuracy

@ray.remote
def ray_DT_eval(model_params: Dict[str, Any], 
                X_train: np.ndarray, 
                y_train: np.ndarray, 
                train_split,
                validation_split, 
                seed: int, id: int) -> Tuple[float, int, float]:
    
    # initialize the RF model
    model = DecisionTreeClassifier(**model_params, random_state=seed)

    # try to fit
    try:
        # fit model on training data split
        model.fit(X_train[train_split], y_train[train_split])
        accuracy = model.score(X_train[validation_split], y_train[validation_split])
        return -1.0 * float(accuracy), id, 1.0, None

    except Exception as e:
        # failed
        return -1.0, id, -1.0, str(e)


###########################################################
#                       KERNEL SVC                        #
###########################################################
@typechecked
def eval_parameters_KSVC_final(model_params: Dict[str, Any], 
                             X_train: np.ndarray, 
                             y_train: np.ndarray, 
                             X_test: np.ndarray, 
                             y_test: np.ndarray, 
                             seed: int) -> Tuple[float, float]:
    model = SVC(**model_params, random_state=seed)
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    return train_accuracy, test_accuracy

@ray.remote
def ray_KSVC_eval(model_params: Dict[str, Any], 
                X_train: np.ndarray, 
                y_train: np.ndarray, 
                train_split,
                validation_split, 
                seed: int, id: int) -> Tuple[float, int, float]:
    
    # initialize the RF model
    model = SVC(**model_params, random_state=seed)

    # try to fit
    try:
        # fit model on training data split
        model.fit(X_train[train_split], y_train[train_split])
        accuracy = model.score(X_train[validation_split], y_train[validation_split])
        return -1.0 * float(accuracy), id, 1.0, None

    except Exception as e:
        # failed
        return -1.0, id, -1.0, str(e)


###########################################################
#                      EXTRA TREES                        #
###########################################################
@typechecked
def eval_parameters_ET_final(model_params: Dict[str, Any], 
                             X_train: np.ndarray, 
                             y_train: np.ndarray, 
                             X_test: np.ndarray, 
                             y_test: np.ndarray, 
                             seed: int) -> Tuple[float, float]:
    model = ExtraTreesClassifier(**model_params, random_state=seed)
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    return train_accuracy, test_accuracy

@ray.remote
def ray_ET_eval(model_params: Dict[str, Any], 
                X_train: np.ndarray, 
                y_train: np.ndarray, 
                train_split,
                validation_split, 
                seed: int, id: int) -> Tuple[float, int, float]:
    
    # initialize the RF model
    model = ExtraTreesClassifier(**model_params, random_state=seed)

    # try to fit
    try:
        # fit model on training data split
        model.fit(X_train[train_split], y_train[train_split])
        accuracy = model.score(X_train[validation_split], y_train[validation_split])
        return -1.0 * float(accuracy), id, 1.0, None

    except Exception as e:
        # failed
        return -1.0, id, -1.0, str(e)


###########################################################
#                     GRADIENT BOOST                      #
###########################################################
@typechecked
def eval_parameters_GB_final(model_params: Dict[str, Any], 
                             X_train: np.ndarray, 
                             y_train: np.ndarray, 
                             X_test: np.ndarray, 
                             y_test: np.ndarray, 
                             seed: int) -> Tuple[float, float]:
    model = GradientBoostingClassifier(**model_params, random_state=seed)
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    return train_accuracy, test_accuracy

@ray.remote
def ray_GB_eval(model_params: Dict[str, Any], 
                X_train: np.ndarray, 
                y_train: np.ndarray, 
                train_split,
                validation_split, 
                seed: int, id: int) -> Tuple[float, int, float]:
    
    # initialize the RF model
    model = GradientBoostingClassifier(**model_params, random_state=seed)

    # try to fit
    try:
        # fit model on training data split
        model.fit(X_train[train_split], y_train[train_split])
        accuracy = model.score(X_train[validation_split], y_train[validation_split])
        return -1.0 * float(accuracy), id, 1.0, None

    except Exception as e:
        # failed
        return -1.0, id, -1.0, str(e)


###########################################################
#                      LINEAR SGD                         #
###########################################################
@typechecked
def eval_parameters_LSGD_final(model_params: Dict[str, Any], 
                             X_train: np.ndarray, 
                             y_train: np.ndarray, 
                             X_test: np.ndarray, 
                             y_test: np.ndarray, 
                             seed: int) -> Tuple[float, float]:
    model = SGDClassifier(**model_params, random_state=seed)
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    return train_accuracy, test_accuracy

@ray.remote
def ray_LSGD_eval(model_params: Dict[str, Any], 
                X_train: np.ndarray, 
                y_train: np.ndarray, 
                train_split,
                validation_split, 
                seed: int, id: int) -> Tuple[float, int, float]:
    
    # initialize the RF model
    model = SGDClassifier(**model_params, random_state=seed)

    # try to fit
    try:
        # fit model on training data split
        model.fit(X_train[train_split], y_train[train_split])
        accuracy = model.score(X_train[validation_split], y_train[validation_split])
        return -1.0 * float(accuracy), id, 1.0, None

    except Exception as e:
        # failed
        return -1.0, id, -1.0, str(e)


###########################################################
#                          OTHER                          #
###########################################################
#https://github.com/automl/ASKL2.0_experiments/blob/84a9c0b3af8f7ac6e2a003d4dea5e6dce97d4315/experiment_scripts/utils.py
@typechecked
def load_task(task_id: int, 
              data_dir: str, 
              preprocess=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

@typechecked
def get_task_info(task_id: int, data_dir: str) -> Tuple[int, int, int]:
    task_list_path = f"{data_dir}/task_list.csv"
    if os.path.exists(task_list_path):
        tasks = pd.read_csv(task_list_path)
        row = tasks[tasks['task_id'] == task_id]
        if row.empty:
            raise ValueError(f"Task ID {task_id} not found in {task_list_path}")
        row = row.iloc[0]
        features, rows, classes = row["features"], row["rows"], row["classes"]
    else:
        print(f'Task list not found in {data_dir}')
        exit(0)
    return int(features), int(rows), int(classes)

# remove any individuals wiht a positive performance
@typechecked
def remove_failed_individuals(population: List[Individual], config: Config) -> List[Individual]:
    """
    Removes individuals with a positive performance from the population.
    This is useful for ensuring that only individuals with negative performance are considered.
    A positive performance indicates that the individual failed during evaluation and is not suitable for selection.
    """
    population = [ind for ind in population if ind.get_performance() <= 0.0]
    if config.debug:
        print(f"Removed individuals with positive performance, new population size: {len(population)}", flush=True)
    return population

# return the best training performance from the population
@typechecked
def get_best_performance(population: List[Individual]) -> float:
    """
    Returns the best training performance from the population.
    """
    if not population:
        raise ValueError("Population is empty, cannot get best training performance.")
    return min([ind.get_performance() for ind in population])

# process the current population and update self.best_performers and self.best_performance
@typechecked
def process_population_for_best(population: List[Individual], 
                                best_performance: float, 
                                best_performers: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Processes the current population and updates self.best_performers and self.best_performance.
    """
    current_best = get_best_performance(population)

    # check if we have found a better performance
    if current_best < best_performance:
        best_performance = current_best
        best_performers = []

    # add all individuals with the current best performance to the best performers
    for ind in population:
        if ind.get_performance() == best_performance:
            best_performers.append(copy.deepcopy(ind.get_params()))

    assert len(best_performers) > 0, "No best performers found in the population."
    return best_performance, best_performers