"""
Script to compile generational data from running
TPEBO with mutation variances 0.45 and 0.5 on task 190146.
Produces "tpec-190146.csv" with the following structure:

    replicate,mutv,generation,evaluation,best,average,median,std

Workflow:
1. compile_190146.py (this file) 
2. plot_190146.R
"""

import pandas as pd
import os
import re


if __name__ == "__main__":
    # may not use this, just laying things out for clarity
    mut_vars = [0.45, 0.5]
    tour_size = 10
    mut_rate = 1
    replicates = 20
    task = 190146

    results_path = './190146/'

    log_re = re.compile(
        r"log_TPEC_mutr1,0_mutv(0,45|0,5)_tour10_([0-9]+)"
        )
    
    compiled = [] # this will hold all the data frames

    for filename in os.listdir(results_path):
        match = log_re.match(filename)
        if not match:
            print(f"Skipping unmatched file: {filename}")
            continue

        mutv, rep = match.groups()
        mutv = float(mutv.replace(',', '.'))
        rep = int(rep)

        # sanity check
        if mutv not in mut_vars:
            print(f"Unexpected mutation variance {mutv} in {filename}")
            continue

        filepath = os.path.join(results_path, filename)
        print(filepath)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading {filename}")
            continue
        
        df["replicate"] = rep
        df["mutv"] = mutv

        compiled.append(df)

    if compiled:
        all_data = pd.concat(compiled, ignore_index=True)
        all_data = all_data[["replicate", "mutv", "generation", 
                             "evaluation", "best", "average", 
                             "median", "std"]]
        all_data.to_csv("tpec-190146.csv", index=False)
    else:
        print("No valid log files found?")