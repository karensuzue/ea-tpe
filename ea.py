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

        self.evaluations = self.config.get_evaluations()
        self.pop_size = self.config.get_pop_size()
        self.tour_size = self.config.get_tour_size()
        self.mut_rate = self.config.get_mut_rate()

        self.param_space = self.config.get_param_space()
        
        self.population: List[Organism] = []
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.config.load_dataset()

        # FOR DEBUGGING:
        self.debug = self.config.get_debug()
        self.hard_eval_count = 0 # evaluations on the true objective

    def get_population(self) -> List[Organism]:
        return self.population
    
    # def append_population(self, orgs: List[Organism]) -> None:
    #     """ Deprecated. """
    #     # Place offspring back in population
    #     if self.config.get_replacement_state():
    #         # Sort in ascending order, lowest fitness is better
    #         self.population.sort(key=lambda o: o.get_fitness())  
    #         # Remove lowest performing individuals in the population
    #         self.population = self.population[:len(self.population) - len(orgs)]
    #         # Replace missing individuals with offspring
    #         self.population += orgs
    #         assert(len(self.population) == self.config.get_pop_size())
    #     else:
    #         self.population += orgs
    
    def init_population(self) -> None:
        """
        Initialize the sample population with random genomes based on the parameter space.
        This method clears any existing samples and generates a new population of organisms with
        randomly sampled hyperparameters.
        """
        self.population.clear() # just in case
        for _ in range(self.pop_size):
            genome = {}
            for param_name, spec in self.param_space.items():
                if spec["type"] == "int":
                    genome[param_name] = random.randint(*spec["bounds"])
                elif spec["type"] == "float":
                    genome[param_name] = random.uniform(*spec["bounds"])
                elif spec["type"] == "cat":
                    genome[param_name] = random.choice(spec["bounds"])
                else:
                    raise ValueError(f"Unknown parameter type: {spec['type']}")
            self.population.append(Organism(genome))
        assert (len(self.population) == self.pop_size)

    def evaluate_org(self, org: Organism) -> None:
        # Must maintain the same seed/random_state across experiments  - set 'random_state' to 0
        model = RandomForestClassifier(**org.get_genome(), random_state=0)
        # Inverted, as TPE expects minimization
        score = -1 * cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy').mean() 
        org.set_fitness(score)

    def select_parents(self, num_parents) -> List[Organism]:
        """ 
        Tournament selection.
        Currently mutation only. 
        In the future, reference Starbase's 'variation_order' function and 
        calculate the number of parents needed to produce 'pop_size' offspring.
        """
        parents = []
        # Assume organisms are evaluated
        fitnesses = np.array([org.get_fitness() for org in self.population])
        for _ in range(num_parents): 
            # Randomly choose a tour_size number of indices
            indices = np.random.choice(len(self.population), self.tour_size, replace=False)
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

    def make_offspring(self, parents: List[Organism]) -> List[Organism]:
        offspring: List[Organism] = []
        for org in parents:
            child = self.mutate_org(org, self.mut_rate)
            self.evaluate_org(child)
            offspring.append(child)

            if self.debug: self.hard_eval_count += 1

        assert(len(offspring) == self.pop_size)
        return offspring

    # Run "default" EA 
    def evolve(self) -> None:
        self.init_population()
        for org in self.population:
            self.evaluate_org(org)
            if self.debug: self.hard_eval_count += 1

        generations = self.evaluations // self.pop_size - 1
        for gen in range(generations):
            self.logger.log_generation(gen, self.pop_size * (gen + 1), self.population, "EA")

            parents = self.select_parents(self.pop_size)
            self.population = self.make_offspring(parents)

        # For the final generation
        self.logger.log_generation(generations, self.pop_size * (generations + 1), self.population, "EA")
        self.logger.log_best(self.population, self.config, "EA")
        self.logger.save(self.config, "EA")

        if self.debug:
            print(f"Hard evaluations: {self.hard_eval_count}")

    def run(self) -> None:
        self.evolve()