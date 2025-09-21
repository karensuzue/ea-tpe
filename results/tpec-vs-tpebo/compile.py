"""
Script to compile results from TPEBO and TPEC RF experiments.
Produces "tpec-vs-tpebo.csv".

Workflow:
1. compile.py 
2. plot.R

"""
from natsort import natsorted
import pandas as pd
import os
import re
import json

# Path structure (TPEBO experiments): './results/{TASK ID}/result_TPEBO_{RUN}.jsonl'
# Path structure (TPEC experiments): './results/{TASK ID}/result_TPEBC_mut{MUT_RATE}_tour{TOUR_SIZE}_{RUN}.jsonl'

# csv structure:
# task_id | method | mut_rate | tour_size | replicate | test_accuracy_score | cv_accuracy_score

run_count = 20 

# function to parse a single file of multiple JSON objects for test score and cv score
# each result.json file has 2 JSON objects
def parse_multiple_json_for_score(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Split at '}{' but keep the braces using a positive lookahead/lookbehind
    json_objects = re.split(r'(?<=\})\s*(?=\{)', content)

    test_scores = []
    cv_scores = []
    for i, obj in enumerate(json_objects):
        try:
            # Fix "False"/"True" stringified booleans if needed
            obj_fixed = re.sub(r'"\s*(True|False)\s*"', lambda m: m.group(1).lower(), obj)

            data = json.loads(obj_fixed)
            test_scores.append(data["test_accuracy_score"])
            cv_scores.append(-data["cv_accuracy_score"])
        except Exception as e:
            print(f"Error parsing object #{i+1}: {e}")
            continue

    return test_scores[0], cv_scores[0] # we only need one of each

def make_tpec_df_per_task(results_path: str, task_id: int, mut_rates: list[int], tour_sizes: list[int]):
    """
    Collects TPEC results for one task and returns a DataFrame.
    """   
    # Path to task id folder
    id_folder = os.path.join(results_path, str(task_id))
    print(id_folder)
    if not os.path.isdir(id_folder): # if a specific id does not exist, skip
        print(f"can't find {task_id}")
        return 

    filename_re = re.compile(r"result_TPEC_mut([0-9,\.]+)_tour([0-9]+)_([0-9]+)\.jsonl")

    records = []
    for filename in os.listdir(id_folder):
        # not TPEC results
        if not filename.startswith("result_TPEC"): continue

        # does filename match?
        m = filename_re.match(filename)
        if not m: continue

        mut_raw, tour_raw, replicate = m.groups()
        mut_rate = float(mut_raw.replace(",", ".")) # convert 1,0 to 1.0
        tour_size = int(tour_raw)
        replicate = int(replicate)
        
        # just in case
        if mut_rate not in mut_rates or tour_size not in tour_sizes:
            continue
        
        filepath = os.path.join(id_folder, filename)
        print(filepath)
        test_score, cv_score = parse_multiple_json_for_score(filepath)

        records.append({
                    "task_id": task_id,
                    "method": "TPEC",
                    "mut_rate": mut_rate,
                    "tour_size": tour_size,
                    "replicate": replicate,
                    "test_score": test_score,
                    "cv_score": cv_score
                })
    df = pd.DataFrame.from_records(records)
    return df    

def make_tpebo_df_per_task(results_path: str, task_id: int):
    """
    Collects TPEBO results for one task and returns a DataFrame.
    """   
    # Path to task id folder
    id_folder = os.path.join(results_path, str(task_id))
    print(id_folder)
    if not os.path.isdir(id_folder): # if a specific id does not exist, skip
        print(f"can't find {task_id}")
        return 

    filename_re = re.compile(r"result_TPEBO_([0-9]+)\.jsonl")

    records = []
    for filename in os.listdir(id_folder):
        # not TPEC results
        if not filename.startswith("result_TPEBO"): continue

        # does filename match?
        m = filename_re.match(filename)
        if not m: continue

        replicate = int(m.groups()[0])
        
        filepath = os.path.join(id_folder, filename)
        print(filepath)
        test_score, cv_score = parse_multiple_json_for_score(filepath)

        records.append({
                    "task_id": task_id,
                    "method": "TPEBO",
                    "mut_rate": None,
                    "tour_size": None,
                    "replicate": replicate,
                    "test_score": test_score,
                    "cv_score": cv_score
                })
    df = pd.DataFrame.from_records(records)
    return df    


if __name__ == "__main__":
    tasks = [2073, 168757, 168784, 190146]
    mut_rates = [0.25, 0.5, 0.75, 1.0]
    tour_sizes = [5, 10, 25, 50]
    results_path = '../../results/'

    tpec_df = pd.DataFrame()
    for task in tasks:
        df = make_tpec_df_per_task(results_path, task, mut_rates, tour_sizes)
        tpec_df = pd.concat([tpec_df, df], ignore_index=True)
    print(tpec_df)
    tpec_df.to_csv("tpec_results.csv")

    tpebo_df = pd.DataFrame()
    for task in tasks:
        df = make_tpebo_df_per_task(results_path, task)
        tpebo_df = pd.concat([tpebo_df, df], ignore_index=True)
    print(tpebo_df)
    tpebo_df.to_csv("tpebo_results.csv")

    merged_df = pd.concat([tpec_df, tpebo_df], ignore_index=True)
    merged_df = merged_df.sort_values(["task_id", "method"])
    merged_df.to_csv("tpec-vs-tpebo.csv", index=False)