import os

def check_slurm_outputs(directory, job_id=2156939, max_id=1080):
    missing_files = []
    incomplete_files = []

    for i in range(1, max_id + 1):
        filename = f"slurm-{job_id}_{i}.out"
        filepath = os.path.join(directory, filename)

        # Check if file exists
        if not os.path.exists(filepath):
            missing_files.append(i)
            continue

        # Check last line for "Elapsed time"
        try:
            with open(filepath, "r") as f:
                lines = f.read().strip().splitlines()
                if not lines or "Elapsed time" not in lines[-1]:
                    incomplete_files.append(i)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            incomplete_files.append(i)

    # Print results
    print("Missing files:", ",".join(map(str, sorted(missing_files))))
    print("Incomplete files:", ",".join(map(str, sorted(incomplete_files))))


if __name__ == "__main__":
    # Replace with your directory path
    directory = "/home/hernandezj45/Repos/ea-tpe"
    check_slurm_outputs(directory)