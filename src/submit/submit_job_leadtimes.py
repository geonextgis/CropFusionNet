import importlib
import os
import subprocess
import sys
import uuid

# --- CONFIGURATION ---
CROPS = ["silage_maize", "winter_barley", "winter_wheat"]
INTERVAL = 10  # days between each lead-time step

source_folder = "/beegfs/halder/GITHUB/RESEARCH/crop-yield-forecasting-germany/src"
sys.path.append(source_folder)

working_dir = "/beegfs/halder/GITHUB/RESEARCH/crop-yield-forecasting-germany/src/train"
base_output_dir = (
    "/beegfs/halder/GITHUB/RESEARCH/crop-yield-forecasting-germany/src/train/leadtimes"
)


def load_config(crop_name: str):
    module_path = f"config.{crop_name}"
    cfg = importlib.import_module(module_path)
    return cfg


def generate_leadtime_steps(
    forecast_scenarios: dict, interval: int
) -> list[tuple[int, str]]:
    """
    Generate (seq_length, label) pairs at every `interval` days between the
    minimum and maximum seq_length defined in forecast_scenarios.

    Returns a list of (seq_length, label) tuples sorted by seq_length.
    """
    seq_lengths = list(forecast_scenarios.values())
    seq_min = min(seq_lengths)
    seq_max = max(seq_lengths)

    steps = []
    seq = seq_min
    while seq <= seq_max:
        label = f"day_{seq:03d}"
        steps.append((seq, label))
        seq += interval

    # Always include the last point if not already added
    if steps[-1][0] != seq_max:
        steps.append((seq_max, f"day_{seq_max:03d}"))

    return steps


# Ensure slurm log directory exists
os.makedirs("slurm_logs", exist_ok=True)

# --- SUBMISSION LOOP: iterate over all crops at 10-day lead-time intervals ---
for crop in CROPS:
    config = load_config(crop)
    forecast_scenarios = config.forecast_scenarios

    output_base_dir = os.path.join(base_output_dir, crop)
    os.makedirs(output_base_dir, exist_ok=True)

    leadtime_steps = generate_leadtime_steps(forecast_scenarios, INTERVAL)
    print(
        f"\n🌾 Submitting {len(leadtime_steps)} lead-time jobs for: {crop} "
        f"(seq_length {leadtime_steps[0][0]} → {leadtime_steps[-1][0]}, every {INTERVAL} days)"
    )

    for seq_length, label in leadtime_steps:
        # 1. Generate Unique ID
        unique_id = str(uuid.uuid4())[:8]

        output_dir = os.path.join(output_base_dir, label)
        os.makedirs(output_dir, exist_ok=True)

        job_name = f"{crop}_lt_{label}_{unique_id}"

        # 2. Construct Python Command
        python_cmd = (
            f"python train_CropFusionNet.py "
            f"--crop {crop} "
            f"--job_id {unique_id} "
            f"--seq_length {seq_length} "
            f"--output_dir {output_dir}"
        )

        # 3. Create SLURM Script Content
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu

# Load modules or activate environment
source ~/.bashrc
conda activate torch

# Move to the correct directory
cd {working_dir}

echo "Starting job {unique_id} | Crop: {crop} | Lead time: {label} (seq_length={seq_length})..."
{python_cmd}
echo "✅ Finished job {unique_id} | Crop: {crop} | Lead time: {label}"
"""

        # 4. Write and Submit Job
        script_filename = f"job_{crop}_{label}_{unique_id}.sh"
        with open(script_filename, "w") as f:
            f.write(slurm_script)

        print(
            f"  ➡ Submitting: {crop} | {label} | seq_length: {seq_length} | ID: {unique_id}"
        )
        subprocess.run(["sbatch", script_filename])

        # Cleanup: Remove the .sh file after submission to keep folder clean
        os.remove(script_filename)

print("\n✅ All lead-time jobs submitted for all crops!")
