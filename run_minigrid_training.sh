#!/bin/bash

### MiniGrid
envs=(
  "MiniGrid-Empty-Random-5x5-v0"
  "MiniGrid-DoorKey-5x5-v0"
  "MiniGrid-Fetch-5x5-N2-v0"
  "MiniGrid-Unlock-v0"
)

# Loop through each environment and run the trainer
for env in "${envs[@]}"; do
    echo "--------------------------------------------------"
    echo "Starting training for: $env"
    echo "--------------------------------------------------"

    python3 ./sb3_agents/trainer.py --emulator minigrid --env "$env"

    echo "Finished training for: $env"
done