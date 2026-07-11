#!/bin/bash

### Gymnasium
envs=(
    "highway-v0"
    "merge-v0"
    "roundabout-v0"
    "parking-v0"
    "intersection-v0"
    "racetrack-v0"
    "lane-keeping-v0"
    "two-way-v0"
    "exit-v0"
    "u-turn-v0"
)

# Loop through each environment and run the trainer
for env in "${envs[@]}"; do
    echo "--------------------------------------------------"
    echo "Starting training for: $env"
    echo "--------------------------------------------------"

    python3 ./sb3_agents/trainer.py --emulator other --env "$env"

    echo "Finished training for: $env"
done