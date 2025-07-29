import ray

# runner.sb calls this function
# 4 methods × 4 task IDs × 15 replicates = 240 total jobs,
def execute_experiment(task_id: int, n_jobs: int, save_path: str, seed: int, data_dir: str):
    start_time = time.time()
 
    # variables
    total_evals = 10000

    # initialize ray
    ray.init(num_cpus=n_jobs, include_dashboard=True)

    # generate directory to save results
