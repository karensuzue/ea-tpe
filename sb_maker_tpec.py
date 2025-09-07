"""
File to generate SLURM script for TPEC
"""

# sbatch --export=OFFSET=$((i * 1024)) runner_tpec.sb

import pandas as pd

def make_header(array_max, nodes = 1, ntasks = 1, cpus = 12):
    lines = [
        "#!/bin/bash",
        "########## Define Resources Needed with SBATCH Lines ##########",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks={ntasks}",

        # Comment out if jobs exceed limit
        # f"#SBATCH --array=1-{array_max}",
        f"#SBATCH --array=1-1024",

        f"#SBATCH --cpus-per-task={cpus}",
        "#SBATCH -t 24:00:00",
        "#SBATCH --mem=100GB",
        "#SBATCH --job-name=tpe",
        "#SBATCH -p defq",
        # "#SBATCH --exclude=esplhpc-cp040",
        "###############################################################\n"
    ]

    return lines

if __name__ == "__main__":
    # Retrieve names of the hardest OpenML tasks (less than 80% accuracy on test set)
    df = pd.read_csv('./data/agg_test.csv')
    df_hard = df[df['avg_test_score'] < 0.8] 
    task_ids = df_hard['task_id'].tolist()

    evals = 1000
    method = 'TPEC'

    # For a total of 45*task_num jobs
    replicates = 20
    tour_sizes = [5, 10, 25, 50]
    mut_rates = [.25, .5, .75, 1.0]
    total_jobs = len(task_ids) * replicates * len(tour_sizes) * len(mut_rates)

    num_cpus = 12

    lines = make_header(array_max = total_jobs, cpus = num_cpus)

    lines.append("source ~/anaconda3/etc/profile.d/conda.sh")
    lines.append("conda init")
    lines.append("conda activate tpe-ea\n")

    lines.append('DATA_DIR=/mnt/home/suzuekar/ea-tpe/data/')
    lines.append("RESULTS_DIR=/mnt/home/suzuekar/ea-tpe/results\n")

    lines.append('##################################')
    lines.append('# Treatments')
    lines.append('##################################\n')

    lines.append("DATASETS=(" + " ".join(str(id) for id in task_ids) + ")")
    lines.append("TOUR_SIZES=(" + " ".join(str(ts) for ts in tour_sizes) + ")")
    lines.append("MUT_RATES=(" + " ".join(str(mr) for mr in mut_rates) + ")\n")

    # slurm array starts at 1, bash array starts at 0
    lines.append("ID=$(( SLURM_ARRAY_TASK_ID - 1 + ${OFFSET:-0} ))")

    lines.append(f"DATASET_ID=$(( ID / ({replicates} * {len(mut_rates)} * {len(tour_sizes)}) ))")
    lines.append(f"TOUR_SIZE_ID=$(( (ID / ({len(mut_rates)} * {replicates})) % {len(tour_sizes)} ))")
    lines.append(f"MUT_RATE_ID=$(( (ID / {replicates}) % {len(mut_rates)} ))")
    lines.append(f"REPLICATE=$(( ID % {replicates} ))\n")

    lines.append("DATASET=${DATASETS[$DATASET_ID]}")
    lines.append("TOUR_SIZE=${TOUR_SIZES[$TOUR_SIZE_ID]}")
    lines.append("MUT_RATE=${MUT_RATES[$MUT_RATE_ID]}\n")

    lines.append("echo \"ID=$ID Dataset=$DATASET TourSize=$TOUR_SIZE MutRate=$MUT_RATE Replicate=$REPLICATE\" \n")

    # lines.append("python /home/hernandezj45/Repos/ea-tpe/main.py \\")
    lines.append("python /mnt/home/suzuekar/ea-tpe/main.py \\")
    lines.append(f"    --method {method} \\")
    lines.append("    --task_id ${DATASET} \\")
    lines.append("    --seed ${REPLICATE} \\")
    lines.append("    --tour_size ${TOUR_SIZE} \\")
    lines.append("    --mut_rate ${MUT_RATE} \\")
    lines.append(f"    --evaluations {evals} \\")
    lines.append(f"    --num_cpus {num_cpus} \\")
    lines.append("    --debug True \\")
    lines.append("    --logdir ${RESULTS_DIR}")

    with open('runner_tpec.sb', "w") as f:
        for line in lines:
            f.write(line + "\n")




