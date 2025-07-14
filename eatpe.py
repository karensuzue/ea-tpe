import numpy as np
from config import Config
from logger import Logger
from ea import EA
from tpe import TPE

class EATPE:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

        self.ea = EA(config, logger) 
        self.tpe = TPE(config, logger)
    
    
    def suggest(self, candidates):
        k = self.config.get_num_child()
        return self.tpe.suggest(candidates, k)

    def evolve(self):
        # Get the number of generations
        generations = self.config.get_generations()

        # Initialize EA population
        self.ea.init_population()
        # Evaluate EA population
        self.ea.evaluate_population()

        for gen in range(generations):
            # Log EA population
            self.logger.log_generation(gen, self.ea.get_population())

            # Set EA population as TPE sample set
            self.tpe.set_samples(self.ea.get_population())

            # Split sample set into 'good' and 'bad' groups
            good_samples, bad_samples = self.tpe.split_samples()

            # Fit TPE
            self.tpe.fit(good_samples, bad_samples)

            # Select parents, produce offspring
            parents = self.ea.select_parents()
            offspring = []
            mut_rate = self.config.get_mut_rate()
            for org in parents:
                child = self.ea.mutate_org(org, mut_rate)
                # self.ea.evaluate_org(child) # to avoid unnecessary evaluations in the main loop
                offspring.append(child)

            # Score offspring on Expected Improvement, choose top-k best
            best_org, _ = self.suggest(offspring)
            for org in best_org: # 'best_org' maybe contain more than 1 organism
                self.ea.evaluate_org(org) 
            
            self.ea.append_population(best_org)

        # For the final generation
        self.logger.log_generation(generations, self.ea.population)
        self.logger.log_best(self.ea.population, self.config, "EA+TPE")
        self.logger.save(self.config, "EA+TPE")





