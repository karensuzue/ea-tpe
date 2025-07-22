import argparse
from config import Config
from logger import Logger
from ea import EA
from tpe import TPE
from eatpe import EATPE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 0,
                        help = "Random seed for reproducibility.")
    parser.add_argument('--evaluations', type = int, default = 500,
                        help = "Total number of evaluations of the objective function.")
    parser.add_argument("--pop_size", type = int, default = 50,
                        help = "Population size.")
    parser.add_argument('--num_candidates', type=int, default=10,
                        help="Number of candidate offspring produced per parent for EA+TPE.")
    parser.add_argument("--tour_size", type = int, default = 5, 
                        help = "Tournament size for selection.")
    parser.add_argument("--mut_rate", type = float, default = 0.1,
                        help = "Mutation rate per gene.")
    parser.add_argument('--dataset', type = int, choices = [0, 1, 2], default = 0, 
                        help = "Index of the dataset to use for hyperparameter tuning.")
    parser.add_argument("--logdir", type = str, default = "results",
                        help = "Directory to store logs and results.")
    parser.add_argument("--debug", type = bool, default = False,
                        help = "For development purposes; runs may take longer when debug mode is enabled.")
    args = parser.parse_args()

    config = Config(
        seed = args.seed,
        evaluations = args.evaluations,
        pop_size = args.pop_size,
        num_candidates = args.num_candidates,
        tour_size = args.tour_size,
        mut_rate  =  args.mut_rate,
        dataset_idx = args.dataset,
        debug = args.debug
    )

    logger = Logger(
        logdir = args.logdir
    )

    ea_solver = EA(config, logger)
    ea_solver.run()

    # tpe_solver = TPE(config, logger)
    # tpe_solver.run()

    # eatpe_solver = EATPE(config, logger)
    # eatpe_solver.run()


if __name__ == "__main__":
    main()
