import numpy as np
import random
from config import ParamSpace, Config
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

class Organism:
    """
    A set of hyperparameters to be evolved.
    """
    def __init__(self, genome: ParamSpace):
        self.fitness: float | None = None
        self.genome = genome

    def __repr__(self):
        return f"Organism(genome={self.genome}, fitness={self.fitness})"
    
    def set_fitness(self, f: float) -> None:
        self.fitness = f
    
    def get_fitness(self) -> float:
        if self.fitness is None:
            raise ValueError("Organism hasn't been evaluated.")
        return self.fitness

    def set_genome(self, g: ParamSpace) -> None:
        self.genome = g
    
    def get_genome(self) -> ParamSpace:
        return self.genome

class EA:
    """
    EA Solver.
    """
    def __init__(self, config: Config):
        self.config = config
        self.population: List[Organism] = []

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

    def evaluate_org(self, org: Organism, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # Must maintain the same seed/random_state across experiments 
        model = RandomForestClassifier(**org.get_genome(), random_state=0)
        # Inverted, as TPE expects minimization
        score = -1 * cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean() 
        org.set_fitness(score)

    def evaluate_population(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        for org in self.population:
            self.evaluate_org(org, X_train, y_train)

    def select_parents(self) -> List[Organism]:
        """ Tournament selection. """
        tour_size = self.config.get_tour_size()
        parents = []
        # Assume organisms are evaluated
        fitnesses = np.array([org.get_fitness() for org in self.population])
        for _ in range(len(self.population)):
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
            offspring.append(child)

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

