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
    
    def evolve(self):
        # Get the number of generations
        generations = self.config.get_generations()
        pop_size = self.config.get_pop_size()
        num_child = self.config.get_num_child()
        mut_rate = self.config.get_mut_rate()

        # Initialize EA population
        self.ea.init_population()
        # Evaluate EA population
        self.ea.evaluate_population()

        for gen in range(generations):
            # Log EA population
            self.logger.log_generation(gen, self.ea.get_population(), "EA+TPE")

            # Set EA population as TPE sample set
            self.tpe.set_samples(self.ea.get_population())

            # Split sample set into 'good' and 'bad' groups
            good_samples, bad_samples = self.tpe.split_samples()

            # Fit TPE
            self.tpe.fit(good_samples, bad_samples)

            # Select parents, produce offspring
            # Has to be more than 'num_child', otherwise there is no point in selecting candidates based on EI
            num_parents = min(num_child * 2, pop_size)
            parents = self.ea.select_parents(num_parents) # 'num_child' * 2, or 'pop_size' parents
            offspring = [] # 'num_child' * 2, or 'pop_size' offspring
            for org in parents:
                child = self.ea.mutate_org(org, mut_rate)
                offspring.append(child)
            
            # Evaluate offspring on expected improvement (TPE surrogate)
            best_org, ei_scores = self.tpe.suggest_num_child(offspring) # 'num_child' candidates

            # Log per-iteration expected improvement statistics
            self.logger.log_ei(gen, ei_scores)

            # Evaluate best candidates on the true objective
            for org in best_org: # 'best_org' maybe contain more than 1 organism
                self.ea.evaluate_org(org) 
            
            self.ea.append_population(best_org)

        # For the final generation
        self.logger.log_generation(generations, self.ea.population, "EA+TPE")
        self.logger.log_best(self.ea.population, self.config, "EA+TPE")
        self.logger.save(self.config, "EA+TPE")





