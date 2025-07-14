import numpy as np
import os
import json
from config import Config
from organism import Organism
from typing import List

class Logger:
    def __init__(self, logdir: str):
        self.logdir = logdir
        self.history = []
        self.best_org = None
        self.best_org_data = {}

    def log_generation(self, generation: int, population: List[Organism]) -> None:
        best = min(population, key=lambda o: o.get_fitness())
        avg = np.mean([o.get_fitness() for o in population])
        median = np.median([o.get_fitness() for o in population])
        self.history.append((generation, best.get_fitness(), avg, median))
        print(f"[LOG] Generation {generation}: {len(population)} organisms")
    
    def log_best(self, population: List[Organism], config: Config, method: str) -> None:
        """ Records the best hyperparameter in the current population. """
        population.sort(key=lambda o: o.get_fitness())
        self.best_org = population[0]
        self.best_org_data = {
            "dataset": config.get_dataset_id(),
            "replicate": config.get_seed(),
            "evaluations": config.get_evaluations(),
            "method": method,
            "final_cv_accuracy_score": self.best_org.get_fitness(),
            "params": self.best_org.get_genome()
        }

    def save(self, config: Config, method: str) -> None:
        dataset = config.get_dataset_id()
        seed = config.get_seed()
        os.makedirs(f"{self.logdir}_dataset{dataset}", exist_ok=True)
        with open(f"{self.logdir}_dataset{dataset}/log_{method}_{seed}.csv", "w") as f:
            f.write("generation,best,average,median\n")
            for gen, best, avg, median in self.history:
                f.write(f"{gen},{best},{avg},{median}\n")

        with open(f"{self.logdir}_dataset{dataset}/result_{method}_{seed}.json", "w") as f:
            json.dump(self.best_org_data, f, indent=2)
    
