import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Choose dataset
dataset_idx = 0 # [0, 1, 2]
results_dir = Path(f"results_dataset{dataset_idx}")


# Choose method and seed
method = "TPE" # "EA+TPE", "TPE", "EA"
seed = 0
log_dir = results_dir / f"log_{method}_{seed}.csv"
log = pd.read_csv(log_dir)

print(log)
print(log.columns)


# STD stats
print("Max std:", log['std'].max())
print("Mean std:", log['std'].mean())
print("Final std:", log['std'].iloc[-1])


plt.plot(log['generation'], log['average'], label='Average Fitness')
plt.plot(log['generation'], log['best'], label='Best Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.title(f'{method}-{seed}: Fitness over time')
plt.grid(True)
plt.show()