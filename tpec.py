import numpy as np
import copy
import ray
from ray import ObjectRef
from config import Config
from logger import Logger
from typing import List, Dict
from typeguard import typechecked
from ea import EA
from tpe import TPE
from individual import Individual
from param_space import ModelParams
from utils import eval_final_factory, remove_failed_individuals, process_population_for_best, evaluation
from sklearn.model_selection import KFold


@typechecked
class TPEC:
    def __init__(self, config: Config, logger: Logger, param_space: ModelParams):
        self.config = config
        self.logger = logger
        self.param_space = param_space

        self.ea = EA(config, logger, param_space)
        self.tpe = TPE() # default gamma (splitting threshold)

        self.population: List[Individual] = []

        # to keep current best Individuals
        self.best_performance = 1.0
        self.best_performers: List[Dict] = []

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
            ray.init(num_cpus = self.config.num_cpus, include_dashboard = False)

        # Initialize population with random organisms
        self.population = [Individual(self.param_space.generate_random_parameters()) for _ in range(self.config.pop_size)]

        # Create Ray ObjectRefs to efficiently share across multiple Ray tasks
        X_train_ref = ray.put(X_train)
        y_train_ref = ray.put(y_train)

        # create cv splits for cross-validation
        cv = KFold(n_splits=self.config.cv_k, shuffle=True, random_state=self.config.seed)
        splits = list(cv.split(X_train, y_train))

        # Evaluate initial population with Ray
        evaluation(self.population, X_train_ref, y_train_ref, splits, self.config.model, self.config.seed)
        if self.config.debug: self.hard_eval_count += len(self.population)

        # Remove individuals with positive performance
        self.population = remove_failed_individuals(self.population, self.config)
        # Update best performance and set of best performers
        self.best_performance, self.best_performers = process_population_for_best(self.population, self.best_performance, self.best_performers)
        print(f"Initial population size: {len(self.population)}", flush=True)
        print(f"Best training performance so far: {self.best_performance}", flush=True)

        # TPE cannot be fitted over numeric parameters with the value "None" (e.g., max_samples),
        # so we create a deep copy of the original archive and assign such parameters a value close to 0.
        # A deep copy is required to preserve the performance of the original individuals,
        # which is necessary for fitting a TPE over the parameter space.
        tpe_archive: List[Individual] = [copy.deepcopy(ind) for ind in self.population]
        for ind in tpe_archive:
            ind.set_params(self.param_space.tpe_parameters(ind.get_params()))
        assert(len(tpe_archive) == len(self.population))

        generations = (self.config.evaluations // self.config.pop_size) - 1
        for gen in range(generations):
            # Log population performance
            self.logger.log_generation(gen, self.config.pop_size * (gen + 1), self.population, "TPEC")

            # Fit TPE to the modified archive of all observed individuals
            self.tpe.fit(tpe_archive, self.param_space, self.config.rng)

            # To store the next population
            new_pop = []

            # List containing the expected improvement scores of the chosen offspring (1 per parent),
            # or essentially, members of the new population
            ei_all_parents: List[float] = []

            # Select enough parents to cover the entire population using tournament selection
            performances = np.array([ind.get_performance() for ind in self.population]) # from the current population
            parents = [self.ea.tournament_selection(self.population, performances) for _ in range(self.config.pop_size)]
            assert(len(parents) == self.config.pop_size)

            # Each parent produces 'num_candidates' candidate offsprings (10 by default)
            for parent in parents:
                # Deep copy parent parameters separately to prevent offspring from being pre-evaluated
                candidate_params = [copy.deepcopy(parent.get_params()) for _ in range(self.config.num_candidates)]
                candidate_params_tpe = [] # copy of 'candidate_params', but modified, leave original as fail-safe
                for param in candidate_params:
                    # Mutations occur in place, includes fixing
                    self.param_space.mutate_parameters(param, self.config.mut_rate, self.config.mut_var)
                    # Can't use TPE's suggest() on candidates if numeric parameters have value "None" (e.g. max_samples)
                    candidate_params_tpe.append(self.param_space.tpe_parameters(param))

                candidates = [Individual(param) for param in candidate_params_tpe]
                assert len(candidates) == self.config.num_candidates

                # Evaluate each candidate's expected improvement score, suggest one
                _, best_index, best_ei_scores, se_count = self.tpe.suggest(self.param_space, candidates, num_top_cand = 1)
                if self.config.debug: self.soft_eval_count += se_count
                assert len(best_index) == 1

                # Append to new population
                new_pop += [Individual(candidate_params[i]) for i in best_index]
                ei_all_parents += best_ei_scores.tolist() # this should be one score

            # Update population
            assert len(new_pop) == self.config.pop_size, f"New population size {len(new_pop)} does not match expected size {self.config.pop_size}."

            # set new population
            self.population = new_pop

            # Evaluate the new population with Ray
            evaluation(self.population, X_train_ref, y_train_ref, splits, self.config.model, self.config.seed)
            if self.config.debug: self.hard_eval_count += len(self.population)

            # remove individuals with positive performance
            self.population = remove_failed_individuals(self.population, self.config)
            # Update best performance and set of best performers
            self.best_performance, self.best_performers = process_population_for_best(self.population, self.best_performance, self.best_performers)
            print(f"Population size at gen {gen}: {len(self.population)}", flush=True)
            print(f"Best training performance so far: {self.best_performance}", flush=True)

            # Log per-iteration expected improvement statistics
            assert(len(ei_all_parents) == self.config.pop_size)
            self.logger.log_ei(gen, self.config.pop_size * (gen + 1), ei_all_parents)

            # Append modified copy of population to the modified archive
            tpe_population: List[Individual] = [copy.deepcopy(ind) for ind in self.population]
            for ind in tpe_population:
                ind.set_params(self.param_space.tpe_parameters(ind.get_params()))
            tpe_archive += tpe_population

            assert (gen + 2) * self.config.pop_size == len(tpe_archive), \
                   f'Expected { (gen + 2) * self.config.pop_size } individuals in TPE archive, but found { len(tpe_archive) }.'

        # randomly select one of the tied best individuals
        assert len(self.best_performers) > 0, "No best performers found in the population."
        best_ind_params = self.config.rng.choice(self.best_performers)

        modified_best_ind_params = self.param_space.tpe_parameters(best_ind_params)

        # Final scores
        train_accuracy, test_accuracy = eval_final_factory(self.config.model, best_ind_params,
                                                           X_train, y_train, X_test, y_test, self.config.seed)
        best_ind = Individual(params=best_ind_params,
                        performance=self.best_performance,
                        train_score=train_accuracy,
                        test_score=test_accuracy)

        self.logger.log_generation(generations, self.config.pop_size * (generations + 1), self.population, "TPEC")
        self.logger.log_best(best_ind, self.config, "TPEC")
        self.logger.save(self.config, "TPEC")
        self.logger.save_tpe_params(self.config, "TPEC", modified_best_ind_params)

        if self.config.debug:
            print(f"Hard evaluations: {self.hard_eval_count}", flush=True)
            print(f"Soft evaluations {self.soft_eval_count}", flush=True)

        ray.shutdown()
