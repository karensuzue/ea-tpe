import numpy as np
import random
from config import Config
from logger import Logger
from organism import Organism
from typing import List
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

class EA:
    """
    EA Solver.
    """
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

        self.population: List[Organism] = []
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.config.load_dataset()

    def get_population(self) -> List[Organism]:
        return self.population
    
    def append_population(self, orgs: List[Organism]) -> None:
        # Place offspring back in population
        if self.config.get_replacement_state():
            # Sort in ascending order, lowest fitness is better
            self.population.sort(key=lambda o: o.get_fitness())  
            # Remove lowest performing individuals in the population
            self.population = self.population[:len(self.population) - len(orgs)]
            # Replace missing individuals with offspring
            self.population += orgs
            assert(len(self.population) == self.config.get_pop_size())
        else:
            self.population += orgs

    
    def init_population(self) -> None:
        self.population.clear() # just in case

        pop_size = self.config.get_pop_size()
        for _ in range(pop_size):
            genome = {}
            for param_name, spec in self.config.get_param_space().items():
                if spec["type"] == "int":
                    genome[param_name] = random.randint(*spec["bounds"])
                elif spec["type"] == "float":
                    genome[param_name] = random.uniform(*spec["bounds"])
                elif spec["type"] == "cat":
                    genome[param_name] = random.choice(spec["bounds"])
                else:
                    raise ValueError(f"Unknown parameter type: {spec['type']}")
            organism = Organism(genome)
            self.population.append(organism)
        assert (len(self.population) == pop_size)

    def evaluate_org(self, org: Organism) -> None:
        # Must maintain the same seed/random_state across experiments 
        model = RandomForestClassifier(**org.get_genome(), random_state=0)
        # Inverted, as TPE expects minimization
        score = -1 * cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy').mean() 
        org.set_fitness(score)

    def evaluate_population(self) -> None:
        for org in self.population:
            self.evaluate_org(org)

    def select_parents(self) -> List[Organism]:
        """ Tournament selection. """
        tour_size = self.config.get_tour_size()
        num_child = self.config.get_num_child()
        parents = []
        # Assume organisms are evaluated
        fitnesses = np.array([org.get_fitness() for org in self.population])
        for _ in range(num_child): # for now, the number of parents selected = number of offspring produced (mutation only)
            # Randomly choose a tour_size number of indices
            indices = np.random.choice(len(self.population), tour_size, replace=False)
            # Extract fitnesses at the chosen indices
            extracted_fitnesses = fitnesses[indices]
            # Get the position of the best (lowest) fitness in tournament (not population-based index)
            best_tour_idx = np.argmin(extracted_fitnesses)
            best_idx = indices[best_tour_idx]
            parents.append(self.population[best_idx])
        return parents
    
    def mutate_org(self, org : Organism, mut_rate: float) -> Organism:
        genome = org.get_genome().copy()
        param_space = self.config.get_param_space()
        # Per-gene mutation
        for name, spec in param_space.items():
            if random.random() < mut_rate:
                if spec["type"] == "int":
                    genome[name] = random.randint(*spec["bounds"])
                elif spec["type"] == "float":
                    genome[name] = random.uniform(*spec["bounds"])
                elif spec["type"] == "cat":
                    genome[name] = random.choice(spec["bounds"])
        organism = Organism(genome)
        return organism

    def mate_population(self, parents: List[Organism]) -> None:
        # Only mutation for now 
        offspring: List[Organism] = []

        # Make offspring
        mut_rate = self.config.get_mut_rate()
        for org in parents:
            child = self.mutate_org(org, mut_rate)
            self.evaluate_org(child) # to avoid unnecessary evaluations in the main loop
            offspring.append(child)
        assert(self.config.get_num_child() == len(offspring))

        # Place offspring back in population
        if self.config.get_replacement_state():
            # Sort in ascending order, lowest fitness is better
            self.population.sort(key=lambda o: o.get_fitness())  
            # Remove lowest performing individuals in the population
            self.population = self.population[:len(self.population) - len(offspring)]
            # Replace missing individuals with offspring
            self.population += offspring
            assert(len(self.population) == self.config.get_pop_size())
        else:
            self.population += offspring

    # Run "default" EA 
    def evolve(self) -> None:
        generations = self.config.get_generations()

        self.init_population()
        self.evaluate_population()
        for gen in range(generations):
            self.logger.log_generation(gen, self.population)
            parents = self.select_parents()
            self.mate_population(parents)

        # For the final generation
        self.logger.log_generation(generations, self.population)
        self.logger.log_best(self.population, self.config, "EA")
        self.logger.save(self.config, "EA")

