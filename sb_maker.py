"""
File to generate SLURM script
"""
# clear; python sb_maker.py > runner.sb

import pandas as pd

def print_header(array_max, nodes = 1, ntasks = 1, cpus = 12):
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

    print("# todo: load conda environment")
    print("source /common/suzuek/miniconda3/etc/profile.d/conda.sh")
    print("conda activate tpe-ea\n")

    print("# todo: define the output and data directory")
    print('DATA_DIR=/common/suzuek/ea-tpe/data/')
    print("RESULTS_DIR=/common/suzuek/ea-tpe/results")
    # print("mkdir -p ${RESULTS_DIR}\n")

if __name__ == "__main__":
    # Retrieve names of the hardest OpenML tasks (less than 80% accuracy on test set)
    # df = pd.read_csv('/common/suzuek/ea-tpe/data/agg_test.csv')
    df = pd.read_csv('C:/Users/dinne/Documents/GitHub/ea-tpe/data/agg_test.csv')
    df_hard = df[df['avg_test_score'] < 0.8] 
    task_ids = df_hard['task_id'].tolist()

    # For a total of 45*task_num jobs
    task_num = len(task_ids)
    evals = 1000
    replicates = 15
    methods = ['EA', 'TPEBO', 'TPEC']
    total_jobs = replicates * len(methods) * task_num

    num_cpus = 12

    print_header(array_max=total_jobs, cpus = num_cpus)

    print('##################################')
    print('# Treatments')
    print('##################################\n')

    print("METHODS=(", end="")
    print(" ".join(str(name) for name in methods), end="") # space-separated method names 
    print(")")

    print("DATASETS=(", end="")
    print(" ".join(str(id) for id in task_ids), end="")  
    print(")\n")

    # slurm array starts at 1, bash array starts at 0
    print("ID=$((SLURM_ARRAY_TASK_ID - 1))")

    print(f"METHOD_ID=$((ID / (15 * {task_num})))")
    print(f"DATASET_ID=$(( (ID % (15 * {task_num})) / 15 ))")
    print(f"REPLICATE=$((ID % 15))\n")

    print("DATASET=${DATASETS[$DATASET_ID]}")
    print("METHOD=${METHODS[$METHOD_ID]} \n")

    print("echo \"Running: dataset=${DATASET}, method=${METHOD}, replicate=${REPLICATE}\"\n")

    # print("OUTPUT_DIR=${RESULTS_DIR}/${METHOD}_${DATASET}_${REPLICATE}")

    print("python main.py \\ ")
    print("    --method ${METHOD} \\ ")
    print("    --task_id ${DATASET} \\ ")
    print("    --seed ${REPLICATE} \\ ")
    print(f"    --evaluations {evals} \\ ")
    print(f"    --num_cpus {num_cpus} \\")
    print("    --logdir ${RESULTS_DIR}")






