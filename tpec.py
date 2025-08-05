import numpy as np
import copy
import ray
from ray import ObjectRef
from config import Config
from logger import Logger
from typing import List
from ea import EA
from tpe import TPE
from individual import Individual
from param_space import ModelParams
from utils import eval_factory, eval_final_factory

class TPEC:
    def __init__(self, config: Config, logger: Logger, param_space: ModelParams):
        self.config = config
        self.logger = logger
        self.param_space = param_space

        self.ea = EA(config, logger, param_space) 
        self.tpe = TPE(self.config.rng) # default gamma (splitting threshold)

        self.population: List[Individual] = []
        self.evaluated_individuals: List[Individual] = [] # archive of every individual evaluated so far

        # For debugging
        self.hard_eval_count = 0 # evaluations on the true objective
        self.soft_eval_count = 0 # evaluations on the surrogate/expected improvement
    
    def run(self, X_train, y_train, X_test, y_test):
        """ 
        Run TPEC with parallelized fitness evaluations using Ray. 

        Parameters:
            X_train, y_train: The training data.
            X_test, y_test: The testing data.
        """
        # Initialize Ray (multiprocessing)
        if not ray.is_initialized():
            ray.init(num_cpus = self.config.num_cpus, include_dashboard = True)

        # Initialize population with random organisms
        self.population = [Individual(self.param_space.generate_random_parameters()) for _ in range(self.config.pop_size)]

        # Evaluate EA population
        # for ind in self.population:
        #     ind.set_performance(eval_factory(self.config.model, ind.get_params(), X_train, y_train))
        #     if self.config.debug: self.hard_eval_count += 1

        # Create Ray ObjectRefs to efficiently share across multiple Ray tasks 
        X_train_ref = ray.put(X_train)
        y_train_ref = ray.put(y_train)

        # Evaluate population with Ray; 1 remote task per individual or set of hyperparameters
        ray_evals: List[ObjectRef] = [ 
            eval_factory.remote(self.config.model, ind.get_params(), X_train_ref, y_train_ref, self.config.seed, i)
            for i, ind in enumerate(self.population) # each ObjectRef leads to a Tuple[float, int]
        ]
        # Process results as they come in
        while len(ray_evals) > 0:
            # ray.wait() returns a list of references to completed results & a list of remaining jobs. 
            # We remove the finished jobs from ray_evals by updating it to be the list of remaining jobs
            # Here we set 'num_returns' to 1, so it returns one finished reference at a time
            done, ray_evals = ray.wait(ray_evals, num_returns = 1)
            # Extract the only result (Tuple[float, int]) from the 'done' list
            performance, index = ray.get(done)[0] 
            self.population[index].set_performance(performance)
            if self.config.debug: self.hard_eval_count += 1

        # Append population to the archive of all evaluated individuals
        self.evaluated_individuals += self.population
        assert(len(self.evaluated_individuals) == len(self.population))

        generations = (self.config.evaluations // self.config.pop_size) - 1
        for gen in range(generations):
            # Log population performance
            self.logger.log_generation(gen, self.config.pop_size * (gen + 1), self.population, "TPEC")

            # Fit TPE to the archive of all observed individuals 
            self.tpe.fit(self.evaluated_individuals, self.param_space)

            # To store the next population
            new_pop = []
            # List containing the expected improvement scores of the chosen offspring (1 per parent),
            # or essentially, members of the new population
            ei_all_parents: List[float] = [] 

            # Select enough parents to cover the entire population using tournament selection
            performances = np.array([ind.get_performance() for ind in self.population])
            parents = [self.ea.tournament_selection(self.population, performances) for _ in range(self.config.pop_size)]

            assert(len(parents) == self.config.pop_size)

            for parent in parents:
                # Each parent produces 'num_candidates' candidate offsprings (10 by default)
                candidates = [copy.deepcopy(parent) for _ in range(self.config.num_candidates)]
                for child in candidates:
                    # Mutations occur in place
                    self.param_space.mutate_parameters(child.get_params(), self.config.mut_rate)

                # Evaluate each candidate's expected improvement score, choose best
                # There should be a single top candidate per parent
                best_ind, best_ei_scores, se_count = self.tpe.suggest(self.param_space, candidates, num_top_cand = 1) 
                if self.config.debug: self.soft_eval_count += se_count
                
                # Evaluate best candidate(s) on the true objective.
                # The system allows 'best_ind' to contain more than one Individual (num_top_cand >= 1),
                # but in TPEC, this is always set to 1. 
                # for ind in best_ind: # should be 1 ind, by default
                #     ind.set_performance(eval_factory(self.config.model, ind.get_params(), X_train, y_train))
                #     if self.config.debug: self.hard_eval_count += 1
                
                ray_child_evals: List[ObjectRef] = [
                    eval_factory.remote(self.config.model, ind.get_params(), X_train_ref, y_train_ref, self.config.seed, i)
                    for i, ind in enumerate(best_ind)
                ]

                while len(ray_child_evals) > 0:
                    done, ray_child_evals = ray.wait(ray_child_evals, num_returns = 1)
                    performance, index = ray.get(done)[0]
                    best_ind[index].set_performance(performance)
                    if self.config.debug: self.hard_eval_count += 1

                # If num_top_cand = 1, one offspring is produced per parent
                new_pop += best_ind # a bit sloppy, but works
                ei_all_parents += best_ei_scores.tolist() # this should be one score
                
            self.population = new_pop
            self.evaluated_individuals += self.population

            # Log per-iteration expected improvement statistics
            self.logger.log_ei(gen, self.config.pop_size * (gen + 1), ei_all_parents)

        # For the final generation
        # ei_last_gen: List[float] = [o.get_ei() for o in self.population]
        self.population.sort(key=lambda o: o.get_performance()) # lowest first
        best_ind = self.population[0]
        self.param_space.fix_parameters(best_ind.get_params())

        # Final scores
        train_accuracy, test_accuracy = eval_final_factory(self.config.model, best_ind.get_params(),
                                                           X_train, y_train, X_test, y_test, self.config.seed)
        best_ind.set_train_score(train_accuracy)
        best_ind.set_test_score(test_accuracy)

        self.logger.log_generation(generations, self.config.pop_size * (generations + 1), self.population, "TPEC")
        self.logger.log_best(best_ind, self.config, "TPEC")
        # self.logger.log_ei(generations, self.config.pop_size * (generations + 1), ei_last_gen)
        self.logger.save(self.config, "TPEC")

        if self.config.debug:
            print(f"Hard evaluations: {self.hard_eval_count}")
            print(f"Soft evaluations {self.soft_eval_count}")

        ray.shutdown()
