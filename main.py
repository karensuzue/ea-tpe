import argparse
import numpy as np
import os
import pickle
from config import Config
from logger import Logger
from ea import EA
from tpe import TPE
from bo import BO
from tpec import TPEC
from param_space import param_space_factory

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 0,
                        help = "Random seed for reproducibility.")
    parser.add_argument('--evaluations', type = int, default = 500,
                        help = "Total number of evaluations of the objective function.")
    parser.add_argument("--pop_size", type = int, default = 50,
                        help = "Population size.")
    parser.add_argument('--num_candidates', type=int, default=10,
                        help="Number of candidate offspring produced per parent for TPEC.")
    parser.add_argument("--tour_size", type = int, default = 5, 
                        help = "Tournament size for selection.")
    parser.add_argument("--mut_rate", type = float, default = 0.1,
                        help = "Mutation rate per gene.")
    parser.add_argument('--task_id', type = int, default = 359959, 
                        help = "OpenML task ID to use for hyperparameter tuning.")
    parser.add_argument("--logdir", type = str, default = "results",
                        help = "Directory to store logs and results.")
    parser.add_argument("--debug", type = bool, default = False,
                        help = "Enable debug mode for development; runs may take longer.")
    parser.add_argument("--method", type = str, choices = ['EA', 'TPEBO', 'TPEC'], default='TPEC',
                        help = "Hyperparameter tuning method to use.")
    parser.add_argument("--model", type = str, choices = ['RF', 'XGB'], default='RF',
                        help = "Model to be used.")
    args = parser.parse_args()

    rng_ = np.random.default_rng(args.seed)
    X_train, y_train, X_test, y_test = load_task(task_id=args.task_id, data_dir="data")
    param_space = param_space_factory(args.model, rng_)
    
    config = Config(
        seed = args.seed,
        evaluations = args.evaluations,
        pop_size = args.pop_size,
        num_candidates = args.num_candidates,
        tour_size = args.tour_size,
        mut_rate  =  args.mut_rate,
        task_id = args.task_id,
        model = args.model,
        debug = args.debug,
        rng = rng_
    )

    logger = Logger(
        logdir = args.logdir
    )

    if args.method == 'EA':
        solver = EA(config, logger, param_space)
    elif args.method == 'TPEBO':
        solver = BO(config, logger, param_space, 'TPE', num_top_cand=1)
    elif args.method == 'TPEC':
        solver = TPEC(config, logger, param_space)
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    solver.run(X_train, y_train)

if __name__ == "__main__":
    main()
