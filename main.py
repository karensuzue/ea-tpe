import argparse
import time
import numpy as np
from config import Config
from logger import Logger
from ea import EA
from bo import BO
from tpec import TPEC
from utils import load_task, param_space_factory

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
    parser.add_argument("--tour_size", type = int, default = 2,
                        help = "Tournament size for selection.")
    parser.add_argument("--mut_rate", type = float, default = 0.1,
                        help = "Mutation rate per gene.")
    parser.add_argument('--task_id', type = int, default = 359959,
                        help = "OpenML task ID to use for hyperparameter tuning.")
    parser.add_argument("--logdir", type = str, default = "results",
                        help = "Directory to store logs and results.")
    parser.add_argument("--debug", type = bool, default = False,
                        help = "Enable debug mode for development; runs may take longer.")
    parser.add_argument("--method", type = str, choices = ['EA', 'TPEBO', 'TPEC'], default = 'TPEC',
                        help = "Hyperparameter tuning method to use.")
    parser.add_argument("--model", type = str, choices = ['RF', 'XGB'], default = 'RF',
                        help = "Model to be used.")
    parser.add_argument("--num_cpus", type = int, default = 1,
                        help = "The number of CPU cores to use for multiprocessing")
    args = parser.parse_args()

    rng_ = np.random.default_rng(args.seed)
    X_train, y_train, X_test, y_test = load_task(task_id = args.task_id, data_dir = "data")
    param_space = param_space_factory(args.model, rng_)

    config = Config(
        seed = args.seed,
        evaluations = args.evaluations,
        pop_size = args.pop_size,
        num_candidates = args.num_candidates,
        tour_size = args.tour_size,
        mut_rate  =  args.mut_rate,
        task_id = args.task_id,
        logdir = args.logdir,
        model = args.model,
        method = args.method,
        debug = args.debug,
        rng = rng_,
        num_cpus = args.num_cpus
    )

    logger = Logger(
        logdir = args.logdir
    )

    if args.method == 'EA':
        solver = EA(config, logger, param_space)
    elif args.method == 'TPEBO':
        solver = BO(config, logger, param_space, 'TPE', num_top_cand = args.pop_size)
    elif args.method == 'TPEC':
        solver = TPEC(config, logger, param_space)
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    start_time = time.time()
    solver.run(X_train, y_train, X_test, y_test)
    print(f"Elapsed time {(time.time() - start_time) / 3600:.4f} hours.")

if __name__ == "__main__":
    main()
