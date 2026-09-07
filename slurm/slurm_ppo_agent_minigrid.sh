#!/bin/bash -x
#SBATCH --job-name=ppo_agent
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=120:00:00

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
bash ~/SB3_Agents/run_minigrid_training.sh

# ═══════════════════════════════════════════════════════════
# Job Complete
# ═══════════════════════════════════════════════════════════
echo "Job finished at: $(date)"

# Deactivate (optional, job ends anyway)
conda deactivate
