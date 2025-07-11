import argparse
from config import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 0,
                        help = "Random seed")
    parser.add_argument('--evaluations', type = int, default = 500,
                        help = "Number of evaluations on the objective function")
    parser.add_argument("--generations", type = int, default = 50, # this is related to evaluations, will revisit once I have the algo down
                        help = "Number of generations")
    parser.add_argument("--pop_size", type = int, default = 50,
                        help = "Population size")
    parser.add_argument("--tour_size", type = int, default = 5, 
                        help = "Tournament size (for tournament selection)")
    parser.add_argument("--mut_rate", type = float, default = 0.1,
                        help = "Mutation rate per gene")
    parser.add_argument("--replacement", type = bool, default = True,
                        help = "Enables new offspring to replace existing organisms" )
    parser.add_argument('--dataset', type = int, choices = [0, 1, 2], default = 0, 
                        help = "Index of dataset to use for hyperparameter tuning")
    parser.add_argument("--logdir", type = str, default = "results/",
                        help = "Directory to store logs/results")
    args = parser.parse_args()

    config = Config(
        dataset_index = args.dataset,
        seed = args.seed,
        generations = args.generations,
        pop_size = args.pop_size,
        tour_size = args.tour_size,
        mut_rate  =  args.mut_rate,
        evaluations = args.evaluations,
        logdir = args.logdir
    )

    X_train, X_test, y_train, y_test = config.load_dataset()

    #best = run_eatpe(search_space, evaluate, generations=args.generations, pop_size=args.pop_size)
    #print("Best configuration found:")
    #print(best)


if __name__ == "__main__":
    main()
