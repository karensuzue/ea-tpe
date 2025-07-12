import numpy as np
import os
from typing import List

class Logger:
    def __init__(self, logdir: str):
        self.logdir = logdir
        self.history = []


    def log(self, generation, population):
        best = min(population, key=lambda o: o.get_fitness())
        avg = np.mean([o.get_fitness() for o in population])
        self.history.append((generation, best.get_fitness(), avg))
        print(f"[LOG] Generation {generation}: {len(population)} organisms")

    def save(self):
        os.makedirs(self.logdir, exist_ok=True)
        with open(f"{self.logdir}/log.csv", "w") as f:
            f.write("generation,best,average\n")
            for gen, best, avg in self.history:
                f.write(f"{gen},{best},{avg}\n")
