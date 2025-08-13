"""
File to generate SLURM script
"""
# clear; python sb_maker.py > runner.sb

import pandas as pd

def make_header(array_max, nodes = 1, ntasks = 1, cpus = 12):
    lines = [
        "#!/bin/bash",
        "########## Define Resources Needed with SBATCH Lines ##########",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --array=1-{array_max}",
        f"#SBATCH --cpus-per-task={cpus}",
        "#SBATCH -t 24:00:00",
        "#SBATCH --mem=100GB",
        "#SBATCH --job-name=tpe",
        "#SBATCH -p defq",
        "#SBATCH --exclude=esplhpc-cp040",
        "###############################################################\n"
    ]

    return lines


if __name__ == "__main__":
    # Retrieve names of the hardest OpenML tasks (less than 80% accuracy on test set)
    df = pd.read_csv('./data/agg_test.csv')
    df_hard = df[df['avg_test_score'] < 0.8]
    task_ids = df_hard['task_id'].tolist()

    # For a total of 45*task_num jobs
    task_num = len(task_ids)
    evals = 1000
    replicates = 20
    methods = ['EA', 'TPEBO', 'TPEC']
    total_jobs = replicates * len(methods) * task_num

    num_cpus = 12

    lines = make_header(array_max=total_jobs, cpus = num_cpus)

    lines.append("source ~/anaconda3/etc/profile.d/conda.sh")
    lines.append("conda activate tpe-ea\n")

    lines.append('DATA_DIR=/home/hernandezj45/Repos/ea-tpe/data/')
    lines.append("RESULTS_DIR=/home/hernandezj45/Repos/ea-tpe/results\n")

    lines.append('##################################')
    lines.append('# Treatments')
    lines.append('##################################\n')

    lines.append("METHODS=(" + " ".join(str(name) for name in methods) + ")")

    lines.append("DATASETS=(" + " ".join(str(id) for id in task_ids) + ")\n")

    # slurm array starts at 1, bash array starts at 0
    lines.append("ID=$((SLURM_ARRAY_TASK_ID - 1))")

    lines.append(f"METHOD_ID=$((ID / ({replicates} * {task_num})))")
    lines.append(f"DATASET_ID=$(( (ID % ({replicates} * {task_num})) / {replicates} ))")
    lines.append(f"REPLICATE=$((ID % {replicates}))\n")

    lines.append("DATASET=${DATASETS[$DATASET_ID]}")
    lines.append("METHOD=${METHODS[$METHOD_ID]} \n")

    lines.append("echo \"Running: dataset=${DATASET}, method=${METHOD}, replicate=${REPLICATE}\"\n")

    lines.append("python main.py \\")
    lines.append("    --method ${METHOD} \\")
    lines.append("    --task_id ${DATASET} \\")
    lines.append("    --seed ${REPLICATE} \\")
    lines.append(f"    --evaluations {evals} \\")
    lines.append(f"    --num_cpus {num_cpus} \\")
    lines.append("    --debug True \\")
    lines.append("    --logdir ${RESULTS_DIR}")

    with open('runner.sb', "w") as f:
        for line in lines:
            f.write(line + "\n")
