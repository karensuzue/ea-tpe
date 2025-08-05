import numpy as np
import os
import json
from config import Config
from individual import Individual
from typing import List

class Logger:
    def __init__(self, logdir: str):
        self.logdir = logdir
        self.history = []
        self.best_ind = None
        self.best_ind_data = {}

        self.ei_history = []

    def log_generation(self, generation: int, evaluation: int, population: List[Individual], method: str) -> None:
        best = min(population, key=lambda o: o.get_performance())
        avg = np.mean([o.get_performance() for o in population])
        median = np.median([o.get_performance() for o in population])
        std = np.std([o.get_performance() for o in population])
        self.history.append((generation, evaluation, best.get_performance(), avg, median, std))
        print(f"[LOG] Evaluation {evaluation} / Generation {generation}: {len(population)} individuals")
    
    def log_best(self, best_ind: Individual, config: Config, method: str) -> None:
        """ Records the best hyperparameter in the current population. """
        self.best_ind = best_ind
        best_params = self.best_ind.get_params()

        # Boolean values can't be written to JSON, so we convert them to type string
        for name, val in best_params.items():
            if isinstance(val, (bool, np.bool_)):
                best_params[name] = str(val)

        self.best_ind_data = {
            "dataset": config.task_id,
            "replicate": config.seed,
            "evaluations": config.evaluations,
            "method": method,
            "cv_accuracy_score": self.best_ind.get_performance(),
            "train_accuracy_score": self.best_ind.get_train_score(),
            "test_accuracy_score": self.best_ind.get_test_score(),
            "params": best_params
        }
    
    def log_ei(self, generation: int, evaluation: int, ei_scores: np.ndarray):
        avg_ei = np.mean(ei_scores)
        max_ei = np.max(ei_scores)
        std_ei = np.std(ei_scores)
        self.ei_history.append((generation, evaluation, avg_ei, max_ei, std_ei))

    def save(self, config: Config, method: str) -> None:
        dataset = config.task_id
        seed = config.seed
        os.makedirs(f"{self.logdir}/{dataset}", exist_ok=True)
        with open(f"{self.logdir}/{dataset}/log_{method}_{seed}.csv", "w") as f:
            f.write("generation,evaluation,best,average,median,std\n")
            for gen, eval, best, avg, median, std in self.history:
                f.write(f"{gen},{eval},{best},{avg},{median},{std}\n")

        with open(f"{self.logdir}/{dataset}/result_{method}_{seed}.json", "w") as f:
            json.dump(self.best_ind_data, f, indent=2)

        if self.ei_history:
            with open(f"{self.logdir}/{dataset}/ei_{method}_{seed}.csv", "w") as f:
                f.write("generation,evaluation,average,max,std\n")
                for gen, eval, avg_ei, max_ei, std_ei in self.ei_history:
                    f.write(f"{gen},{eval},{avg_ei},{max_ei},{std_ei}\n")
    
