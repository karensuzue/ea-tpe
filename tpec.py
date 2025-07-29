import numpy as np
import copy
from config import Config
from logger import Logger
from typing import List
from ea import EA
from tpe import TPE
from organism import Organism
from param_space import ModelParams

class TPEC:
    def __init__(self, config: Config, logger: Logger, param_space: ModelParams):
        self.config = config
        self.logger = logger
        self.param_space = param_space

        self.ea = EA(config, logger, param_space) 
        self.tpe = TPE() # default gamma (splitting threshold)

        self.population: List[Organism] = []

        # Python returns references, unless 'copy.deepcopy()' is stated
        # self.population = self.ea.get_population()

        # For debugging
        self.hard_eval_count = 0 # evaluations on the true objective
        self.soft_eval_count = 0 # evaluations on the surrogate/expected improvement
    
    def run(self, X_train, y_train):
        # Initialize population with random organisms
        self.population = [Organism(self.param_space) for _ in range(self.config.pop_size)]

        # Evaluate EA population
        for org in self.population:
            org.set_fitness(-1 * self.param_space.eval_parameters(org.get_genotype(), X_train, y_train))
            if self.config.debug: self.hard_eval_count += 1

        generations = (self.config.evaluations // self.config.pop_size) - 1
        for gen in range(generations):
            # Log population fitness
            self.logger.log_generation(gen, self.config.pop_size * (gen + 1), self.population, "TPEC")

            # Fit TPE to the population (observed samples)
            self.tpe.fit(self.population, self.param_space)

            # Select 'pop_size' parents 
            # parents = self.ea.select_parents(self.config.pop_size)
            # parents = copy.deepcopy(self.population)

            new_pop = []
            # List containing the expected improvement scores of the chosen offsprings
            ei_all_parents: List[float] = [] 
            for parent in self.population:
                # Each parent produces 'num_candidates' candidate offsprings (10 by default)
                candidates = [copy.deepcopy(parent) for _ in range(self.config.num_candidates)]
                for child in candidates:
                    # Mutations occur in place
                    self.param_space.mutate_parameters(child.get_genotype(), self.config.mut_rate)

                # Evaluate each candidate's expected improvement score, choose best
                # There should be a single top candidate per parent
                best_org, best_ei_scores, se_count = self.tpe.suggest(self.param_space, candidates, num_top_cand=1) 
                if self.config.debug: self.soft_eval_count += se_count
                
                # Evaluate best candidate(s) on the true objective.
                # The system allows 'best_org' to contain more than one organism (num_top_cand >= 1),
                # but in TPEC, this is always set to 1. 
                for org in best_org: # should be 1 org, by default
                    org.set_fitness(-1 * self.param_space.eval_parameters(org.get_genotype(), X_train, y_train))
                    if self.config.debug: self.hard_eval_count += 1
                
                # If num_top_cand = 1, one offspring is produced per parent
                new_pop += best_org # a bit sloppy, but works
                ei_all_parents += best_ei_scores.tolist() # this should be one score
                
            self.population = new_pop
            # Log per-iteration expected improvement statistics
            self.logger.log_ei(gen, self.config.pop_size * (gen + 1), ei_all_parents)

        # For the final generation
        self.population.sort(key=lambda o: o.get_fitness()) # lowest first
        best_org = self.population[0]
        self.param_space.fix_parameters(best_org.get_genotype())

        self.logger.log_generation(generations, self.config.pop_size * (generations + 1), self.population, "TPEC")
        self.logger.log_best(best_org, self.config, "TPEC")
        self.logger.save(self.config, "TPEC")

        if self.config.debug:
            print(f"Hard evaluations: {self.hard_eval_count}")
            print(f"Soft evaluations {self.soft_eval_count}")


