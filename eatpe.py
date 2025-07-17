import numpy as np
from config import Config
from logger import Logger
from typing import List
from ea import EA
from tpe import TPE
from organism import Organism

class EATPE:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

        self.evaluations = self.config.get_evaluations()
        self.num_candidates = self.config.get_num_candidates()
        self.pop_size = self.config.get_pop_size()
        self.mut_rate = self.config.get_mut_rate()

        self.ea = EA(config, logger) 
        self.tpe = TPE(config, logger)

        # Python returns references, unless 'copy.deepcopy()' is stated
        self.population = self.ea.get_population()

        # FOR DEBUGGING:
        self.debug = self.config.get_debug()
        self.hard_eval_count = 0 # evaluations on the true objective
        self.soft_eval_count = 0 # evaluations on the surrogate/expected improvement
    
    def evolve(self):
        # Initialize EA population (should also update 'self.population', due to Python's use of references)
        self.ea.init_population()
        assert(len(self.population) != 0)

        # Evaluate EA population
        for org in self.population:
            self.ea.evaluate_org(org)
            if self.debug: self.hard_eval_count += 1

        generations = self.evaluations // self.pop_size - 1
        for gen in range(generations):
            # Log population fitness
            self.logger.log_generation(gen, self.pop_size * (gen + 1), self.population, "EA+TPE")

            # Set EA population as TPE sample set
            self.tpe.set_samples(self.population)

            # Split sample set into 'good' and 'bad' groups
            good_samples, bad_samples = self.tpe.split_samples()

            # Fit TPE
            self.tpe.fit(good_samples, bad_samples)

            # Select 'pop_size' parents 
            parents = self.ea.select_parents(self.pop_size)

            self.population.clear()
            # List containing the expected improvement scores of the chosen offsprings
            ei_all_parents: List[float] = [] 
            for p in parents:
                # Each parent produces 'num_candidates' candidate offsprings (10 by default)
                candidates = [self.ea.mutate_org(p, self.mut_rate) for _ in range(self.num_candidates)]

                # Evaluate each candidate's expected improvement score, choose best
                best_org, best_ei_scores = self.tpe.suggest(candidates) # this should be a single offspring and its score
                if self.debug: self.soft_eval_count = self.tpe.soft_eval_count

                # Evaluate best candidate(s) on the true objective.
                # The system allows 'best_org' to contain more than one organism (k >= 1),
                # but in practice, there's usually only one (k = 1). 
                # This is essentially a tentative feature.
                for org in best_org: # should be 1 org, by default
                    self.ea.evaluate_org(org) 
                    if self.debug: self.hard_eval_count += 1
                
                # If k = 1, one offspring is produced per parent
                self.population += best_org # a bit sloppy, but works for now
                ei_all_parents += best_ei_scores # this should be one score

            # Log per-iteration expected improvement statistics
            self.logger.log_ei(gen, self.pop_size * (gen + 1), ei_all_parents)

        # For the final generation
        self.logger.log_generation(generations, self.pop_size * (generations + 1), self.population, "EA+TPE")
        self.logger.log_best(self.population, self.config, "EA+TPE")
        self.logger.save(self.config, "EA+TPE")

        print(f"Hard evaluations: {self.hard_eval_count}")
        print(f"Soft evaluations {self.soft_eval_count}")

    def run(self):
        self.evolve()


