# RL Agents

This project focuses on generating datasets and training Reinforcement Learning agents on various Atari, Retro, and classic environments using Stable Baselines 3.

![Preview](./img/PPO_Agent.gif)

## Project Structure

The project is organized as follows:

- **`trainer.py`**: The main entry point for training agents using the PPO algorithm. It supports configuration for both standard [Atari environments](https://ale.farama.org/environments/) (via `ale-py`) and [Retro games](https://stable-retro.farama.org/) (via `stable-retro`), with [Weights & Biases (WanDB)](https://wandb.ai/markub/ppo-sb3) integration for experiment tracking.
- **`inference.py`**: Script for evaluating trained models and generating datasets. It loads the best model for a given environment, runs inference, and saves the resulting trajectories (states, actions, rewards) into `.npz` files.
- **`save/`**: Contains the best-performing model checkpoints (`best_model.zip`) organized by environment name (e.g., `BreakoutNoFrameskip-v4`, `SonicTheHedgehog2-Genesis-v0`).

## Features

- **PPO Implementation**: Leverages [**Proximal Policy Optimization**](https://arxiv.org/abs/1707.06347) from [stable-baselines3](https://stable-baselines3.readthedocs.io).
- **Environment Support**: Compatible with a wide range of:
  - **Atari Games**: `Boxing`, `Breakout`, `Pong`, etc.
  - **Retro Games**: `SonicTheHedgehog2`, `SuperMarioBros3`, etc.
- **Experiment Tracking**: Built-in integration with [**Weights & Biases**](https://wandb.ai/markub/ppo-sb3) for real-time monitoring of training metrics.
- **Dataset Creation**: Capabilities to record and export high-quality agent gameplay data.
