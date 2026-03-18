"""
Submit SLURM jobs for CropFusionNet ablation study.

Runs 7 scenarios (full model + 6 single-component removals) × 3 crops = 21 jobs.
Results are saved under:
    src/train/ablation/{crop}/{scenario}/
        ├── model_{job_id}.pt
        ├── result_{job_id}.json        ← metrics + ablation flags (for plotting)
        ├── train_outputs.pkl
        ├── validation_outputs.pkl
        └── test_outputs.pkl
"""

import os
import subprocess
import uuid

# =============================================================================
# CONFIGURATION
# =============================================================================
CROPS = ["silage_maize", "winter_barley", "winter_wheat"]

# Each scenario: (name, dict of flags to OVERRIDE — unlisted flags stay True)
ABLATION_SCENARIOS = [
    ("full_model", {}),
    ("no_vsn", {"use_vsn": 0}),
    ("no_temporal_conv", {"use_temporal_conv": 0}),
    ("no_lstm", {"use_lstm": 0}),
    ("no_attention", {"use_attention": 0}),
    ("no_static_enrichment", {"use_static_enrichment": 0}),
    ("no_pyramidal_pooling", {"use_pyramidal_pooling": 0}),
]

PROJECT_ROOT = "/beegfs/halder/GITHUB/RESEARCH/crop-yield-forecasting-germany"
WORKING_DIR = os.path.join(PROJECT_ROOT, "src", "train")
BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "train", "ablation")
TRAIN_SCRIPT = "train_CropFusionNet_ablation.py"

# =============================================================================
# HELPERS
# =============================================================================
ALL_FLAGS = [
    "use_vsn",
    "use_temporal_conv",
    "use_lstm",
    "use_attention",
    "use_static_enrichment",
    "use_pyramidal_pooling",
]


def build_flag_args(overrides: dict) -> str:
    """Build CLI args string: all flags default to 1, overrides set to 0."""
    parts = []
    for flag in ALL_FLAGS:
        value = overrides.get(flag, 1)
        parts.append(f"--{flag} {value}")
    return " ".join(parts)


# =============================================================================
# SUBMISSION
# =============================================================================
os.makedirs("slurm_logs", exist_ok=True)

total_jobs = len(CROPS) * len(ABLATION_SCENARIOS)
job_counter = 0

print(
    f"🔬 Ablation study: {len(ABLATION_SCENARIOS)} scenarios × {len(CROPS)} crops = {total_jobs} jobs\n"
)

for crop in CROPS:
    for scenario_name, flag_overrides in ABLATION_SCENARIOS:
        job_counter += 1
        unique_id = str(uuid.uuid4())[:8]

        output_dir = os.path.join(BASE_OUTPUT_DIR, crop, scenario_name)
        os.makedirs(output_dir, exist_ok=True)

        job_name = f"abl_{crop}_{scenario_name}_{unique_id}"
        flag_args = build_flag_args(flag_overrides)

        # Construct Python command
        python_cmd = (
            f"python {TRAIN_SCRIPT} "
            f"--crop {crop} "
            f"--job_id {unique_id} "
            f"--ablation_name {scenario_name} "
            f"--output_dir {output_dir} "
            f"{flag_args}"
        )

        # SLURM script
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu

# Load environment
source ~/.bashrc
conda activate torch

# Move to the training directory
cd {WORKING_DIR}

echo "========================================"
echo "Ablation Study Job"
echo "  Job ID:    {unique_id}"
echo "  Crop:      {crop}"
echo "  Scenario:  {scenario_name}"
echo "  Disabled:  {flag_overrides if flag_overrides else 'None (full model)'}"
echo "  Output:    {output_dir}"
echo "========================================"

{python_cmd}

echo "✅ Finished: {crop} / {scenario_name} ({unique_id})"
"""

        # Write temp script, submit, clean up
        script_filename = f"job_{crop}_{scenario_name}_{unique_id}.sh"
        with open(script_filename, "w") as f:
            f.write(slurm_script)

        print(
            f"  [{job_counter:2d}/{total_jobs}] {crop:16s} | {scenario_name:24s} | ID: {unique_id}"
        )
        subprocess.run(["sbatch", script_filename])

        os.remove(script_filename)

    print()  # blank line between crops

print(f"✅ All {total_jobs} ablation jobs submitted!")
print(f"📂 Results will be saved under: {BASE_OUTPUT_DIR}/")
