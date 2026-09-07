#!/bin/bash -x
#SBATCH --job-name=ppo_agent
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=90:00:00

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate ai_env

# ═══════════════════════════════════════════════════════════
# Job Information
# ═══════════════════════════════════════════════════════════
echo "Job ID:    $SLURM_JOB_ID"
echo "Job Name:  $SLURM_JOB_NAME"
echo "Node:      $(hostname)"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "═══════════════════════════════════════════════════════════"

# Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ═══════════════════════════════════════════════════════════
# Your Code
# ═══════════════════════════════════════════════════════════
export WANDB_API_KEY=eebd20787e3c9bbde2069bb042f2c1e1e94c1bcd
export HF_TOKEN="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"

bash ~/SB3_Agents/run_atari_training.sh

# ═══════════════════════════════════════════════════════════
# Job Complete
# ═══════════════════════════════════════════════════════════
echo "Job finished at: $(date)"

# Deactivate (optional, job ends anyway)
conda deactivate
