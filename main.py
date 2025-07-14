import argparse
from config import Config
from logger import Logger
from ea import EA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 0,
                        help = "Random seed for reproducibility.")
    parser.add_argument('--evaluations', type = int, default = 500,
                        help = "Total number of evaluations of the objective function (overrides generations if set).")
    # parser.add_argument("--generations", type = int, default = 50,
                        # help = "Number of generations (overriden if evaluations is set).")
    parser.add_argument('--num_child', type=int, default=25,
                        help = "Number of offspring produced per iteration.")
    parser.add_argument("--pop_size", type = int, default = 50,
                        help = "Population size.")
    parser.add_argument("--tour_size", type = int, default = 5, 
                        help = "Tournament size for selection.")
    parser.add_argument("--mut_rate", type = float, default = 0.1,
                        help = "Mutation rate per gene.")
    parser.add_argument("--replacement", type = bool, default = True,
                        help = "Whether offspring replace existing individuals in the population." )
    parser.add_argument('--dataset', type = int, choices = [0, 1, 2], default = 0, 
                        help = "Index of the dataset to use for hyperparameter tuning.")
    parser.add_argument("--logdir", type = str, default = "results",
                        help = "Directory to store logs and results.")
    args = parser.parse_args()

    config = Config(
        dataset_idx = args.dataset,
        seed = args.seed,
        # generations = args.generations,
        # Calculate the number of generations from evaluations or fitness computations
        # We assume that each evaluation happens once - however, they are not "unique"
        # Duplicates of previously evaluated genomes can reoccur in the population
        # If caching is involved, the number of actual evaluations will be less than the number inputted here
        generations = args.evaluations // args.pop_size if args.replacement is True 
                        else (args.evaluations - args.pop_size) // args.num_child,
        num_child = args.num_child,
        pop_size = args.pop_size,
        tour_size = args.tour_size,
        mut_rate  =  args.mut_rate,
        evaluations = args.evaluations,
        replacement = args.replacement,
        # logdir = args.logdir # move this to Logger
    )

    logger = Logger(
        logdir = args.logdir
    )

    ea_solver = EA(config, logger)
    ea_solver.evolve()

    # tpe_solver = TPE(config, logger)
    # tpe_solver.optimize()

    # eatpe_solver = EATPE(config, logger)
    # eatpe_solver.evolve()


if __name__ == "__main__":
    main()
