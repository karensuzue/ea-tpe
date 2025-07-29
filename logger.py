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

        self.ei_history = []

    def log_generation(self, generation: int, evaluation: int, population: List[Organism], method: str) -> None:
        best = min(population, key=lambda o: o.get_fitness())
        avg = np.mean([o.get_fitness() for o in population])
        median = np.median([o.get_fitness() for o in population])
        std = np.std([o.get_fitness() for o in population])
        self.history.append((generation, evaluation, best.get_fitness(), avg, median, std))
        print(f"[LOG] Evaluation {evaluation} / Generation {generation}: {len(population)} organisms")
    
    def log_best(self, best_org: Organism, config: Config, method: str) -> None:
        """ Records the best hyperparameter in the current population. """
        self.best_org = best_org
        best_genotype = self.best_org.get_genotype()

        # Boolean values can't be written to JSON, so we convert them to type string
        for name, val in best_genotype.items():
            if isinstance(val, (bool, np.bool_)):
                best_genotype[name] = str(val)

        self.best_org_data = {
            "dataset": config.task_id,
            "replicate": config.seed,
            "evaluations": config.evaluations,
            "method": method,
            "final_cv_accuracy_score": self.best_org.get_fitness(),
            "params": best_genotype
        }
    
    def log_ei(self, generation: int, evaluation: int, ei_scores: np.ndarray):
        avg_ei = np.mean(ei_scores)
        max_ei = np.max(ei_scores)
        std_ei = np.std(ei_scores)
        self.ei_history.append((generation, evaluation, avg_ei, max_ei, std_ei))

    def save(self, config: Config, method: str) -> None:
        dataset = config.task_id
        seed = config.seed
        os.makedirs(f"{self.logdir}_dataset{dataset}", exist_ok=True)
        with open(f"{self.logdir}_dataset{dataset}/log_{method}_{seed}.csv", "w") as f:
            f.write("generation,evaluation,best,average,median,std\n")
            for gen, eval, best, avg, median, std in self.history:
                f.write(f"{gen},{eval},{best},{avg},{median},{std}\n")

        with open(f"{self.logdir}_dataset{dataset}/result_{method}_{seed}.json", "w") as f:
            json.dump(self.best_org_data, f, indent=2)

        if self.ei_history:
            with open(f"{self.logdir}_dataset{dataset}/ei_{method}_{seed}.csv", "w") as f:
                f.write("generation,evaluation,average,max,std\n")
                for gen, eval, avg_ei, max_ei, std_ei in self.ei_history:
                    f.write(f"{gen},{eval},{avg_ei},{max_ei},{std_ei}\n")
    
