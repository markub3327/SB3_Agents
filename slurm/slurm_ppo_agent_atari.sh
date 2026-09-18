#!/bin/bash

#SBATCH --job-name=ppo_agent
#SBATCH --account=perun2601343
#SBATCH --qos=perun2601343
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai_env

# ===================================================================
# Job information
# ===================================================================
echo "Job ID:          $SLURM_JOB_ID"
echo "Job Name:        $SLURM_JOB_NAME"
echo "Node:            $(hostname)"
echo "Node list:       $SLURM_JOB_NODELIST"
echo "Tasks per node:  $SLURM_NTASKS_PER_NODE"
echo "GPUs per node:   $SLURM_GPUS_ON_NODE"
echo "Nodes:           $SLURM_NNODES"
echo "Total GPUs:      $((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
echo "Machine rank:    $SLURM_PROCID"
echo "==================================================================="

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ===================================================================
# Your code
# ===================================================================
export WANDB_API_KEY=eebd20787e3c9bbde2069bb042f2c1e1e94c1bcd
export HF_TOKEN="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"

nvidia-smi

### Atari 2600
envs=(
    "AssaultNoFrameskip-v4"
    "AtlantisNoFrameskip-v4"
    "BankHeistNoFrameskip-v4"
    "BoxingNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "CrazyClimberNoFrameskip-v4"
    "DefenderNoFrameskip-v4"
    "DemonAttackNoFrameskip-v4"
    "DoubleDunkNoFrameskip-v4"
    "EnduroNoFrameskip-v4"
    "FishingDerbyNoFrameskip-v4"
    "FreewayNoFrameskip-v4"
    "GopherNoFrameskip-v4"
    "JamesbondNoFrameskip-v4"
    "KangarooNoFrameskip-v4"
    "KrullNoFrameskip-v4"
    "KungFuMasterNoFrameskip-v4"
    "PhoenixNoFrameskip-v4"
    "PongNoFrameskip-v4"
    "QbertNoFrameskip-v4"
    "RoadRunnerNoFrameskip-v4"
    "StarGunnerNoFrameskip-v4"
    "TutankhamNoFrameskip-v4"
    "UpNDownNoFrameskip-v4"
    "VideoPinballNoFrameskip-v4"
)

# Loop through each environment and run the trainer
for env in "${envs[@]}"; do
    echo "--------------------------------------------------"
    echo "Starting training for: $env"
    echo "--------------------------------------------------"

    python3 ./sb3_agents/trainer.py --emulator ale --env "$env"

    echo "Finished training for: $env"
done

# ===================================================================
# Job finished
# ===================================================================
echo "Job finished at: $(date)"
conda deactivate