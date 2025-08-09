"""
File to generate SLURM script for BO 
"""
# clear; python sb_maker_bo.py > runner_bo.sb

import pandas as pd

def print_header(array_max, nodes = 1, ntasks = 1, cpus = 2):
    # header prints
    print("#!/bin/bash")
    print("########## Define Resources Needed with SBATCH Lines ##########")
    print(f"#SBATCH --nodes={nodes}")
    print(f"#SBATCH --ntasks={ntasks}")
    print(f"#SBATCH --array=1-{array_max}")
    print(f"#SBATCH --cpus-per-task={cpus}")
    print("#SBATCH -t 24:00:00")
    print("#SBATCH --mem=100GB")
    print("#SBATCH --job-name=tpec_rf")
    print("#SBATCH -p defq")
    print("#SBATCH --exclude=esplhpc-cp040")
    print("###############################################################\n")


if __name__ == "__main__":
    # Retrieve names of the hardest OpenML tasks (less than 80% accuracy on test set)
    # df = pd.read_csv('/common/suzuek/ea-tpe/data/agg_test.csv')
    df = pd.read_csv('C:/Users/dinne/Documents/GitHub/ea-tpe/data/agg_test.csv')
    df_hard = df[df['avg_test_score'] < 0.8] 
    task_ids = df_hard['task_id'].tolist()

    # For a total of 45*task_num jobs
    task_num = len(task_ids)
    evals = 1000
    replicates = 20
    method = 'TPEBO'
    total_jobs = replicates * task_num

    num_cpus = 2

    print_header(array_max=total_jobs, cpus = num_cpus)

    print("# todo: load conda environment")
    print("source /common/suzuek/miniconda3/etc/profile.d/conda.sh")
    print("conda activate tpe-ea\n")

    print("# todo: define the output and data directory")
    print('DATA_DIR=/common/suzuek/ea-tpe/data/')
    print("RESULTS_DIR=/common/suzuek/ea-tpe/results\n")

    print('##################################')
    print('# Treatments')
    print('##################################\n')

    print(f"METHOD={method}")

    print("DATASETS=(", end="")
    print(" ".join(str(id) for id in task_ids), end="")  
    print(")\n")

    # slurm array starts at 1, bash array starts at 0
    print("ID=$((SLURM_ARRAY_TASK_ID - 1))")

    print(f"DATASET_ID=$(( ID / {replicates} ))")
    print(f"REPLICATE=$((ID % {replicates}))\n")

    print("DATASET=${DATASETS[$DATASET_ID]}")

    print("echo \"Running: dataset=${DATASET}, method=${METHOD}, replicate=${REPLICATE}\"\n")

    print("python main.py \\")
    print("    --method ${METHOD} \\")
    print("    --task_id ${DATASET} \\")
    print("    --seed ${REPLICATE} \\")
    print(f"    --evaluations {evals} \\")
    print(f"    --num_cpus {num_cpus} \\")
    print("    --debug True \\")
    print("    --logdir ${RESULTS_DIR}")






