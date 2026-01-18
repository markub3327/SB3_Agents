#!/bin/bash

### Gymnasium
envs=(
    "LunarLander-v3"
    "Taxi-v3"
    "FrozenLake-v1"
    "Acrobot-v1"
    "CartPole-v1"
    "MountainCar-v0"
)

# Loop through each environment and run the trainer
for env in "${envs[@]}"; do
    echo "--------------------------------------------------"
    echo "Starting training for: $env"
    echo "--------------------------------------------------"

    python3 ./sb3_agents/trainer.py --emulator classic --env "$env"

    echo "Finished training for: $env"
done