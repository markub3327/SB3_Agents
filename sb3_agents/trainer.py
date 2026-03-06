#!/usr/bin/env python3
# coding: utf-8

# In[1]:


import argparse

import minigrid
import mars_explorer
import ale_py
import gymnasium
import stable_retro as retro
from gymnasium.wrappers import TimeLimit
from schedule import cosine_schedule
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     MaxAndSkipEnv, WarpFrame)
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (SubprocVecEnv, VecFrameStack,
                                              VecTransposeImage, VecNormalize)
from wandb.integration.sb3 import WandbCallback
from utils import load_hyperparams

import wandb
import torch

gymnasium.register_envs(ale_py)

# In[2]:


def make_retro_env(env_name):
    def _body():
        """
        Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
        """
        env = retro.make(env_name, retro.State.DEFAULT, render_mode="rgb_array")
        env = TimeLimit(env, max_episode_steps=8192)
        env = Monitor(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env, width=96, height=96)
        env = ClipRewardEnv(env)
        return env

    return _body


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning project training PPO agents on Atari, Retro (Sonic, Mario) or classic environments using Stable Baselines 3, featuring dataset generation and WanDB experiment tracking."
    )
    parser.add_argument(
        "--emulator",
        type=str,
        required=True,
        choices=["ale", "retro", "classic", "minigrid"],
        help="The name of the emulator ['ale', 'retro', 'classic']",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="The name of the environment (e.g., 'BreakoutNoFrameskip-v4', 'SuperMarioBros3-Nes-v0', 'LunarLander-v3')",
    )
    args = parser.parse_args()

    # Initialize WanDB
    run = wandb.init(
        project="ppo-sb3",
        config={"env_name": args.env},
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    # For Atari console
    if "ale" in args.emulator.lower():
        # Load PPO configuration
        config = load_hyperparams(args.emulator)
        # Create environment
        vec_env = make_atari_env(args.env, n_envs=config["n_envs"], seed=1234)
        vec_env = VecFrameStack(vec_env, n_stack=config["frame_stack"])
        vec_env = VecTransposeImage(vec_env)
    # For stable-retro consoles
    elif "retro" in args.emulator.lower():
        # Load PPO configuration
        config = load_hyperparams(args.emulator)
        # Create environment
        vec_env = SubprocVecEnv([make_retro_env(args.env)] * config["n_envs"])
        vec_env = VecFrameStack(vec_env, n_stack=config["frame_stack"])
        vec_env = VecTransposeImage(vec_env)
        vec_env.action_space.seed(1234)
        vec_env.seed(1234)
    # For MiniGrid
    elif "minigrid" in args.emulator.lower():
        # Load PPO configuration
        config = load_hyperparams(args.env)
        # Create environment
        vec_env = make_vec_env(args.env, n_envs=config["n_envs"], seed=1234)
        # Use MiniGrid wrapper
        vec_env = minigrid.wrappers.FlatObsWrapper(vec_env)
    # For Classic
    elif "classic" in args.emulator.lower():
        # Load PPO configuration
        config = load_hyperparams(args.env)
        # Create environment
        vec_env = make_vec_env(args.env, n_envs=config["n_envs"], seed=1234)
        if config["policy"] == "CnnPolicy":
            vec_env = VecTransposeImage(vec_env)
    else:
        raise ValueError(f"Unsupported emulator: {args.emulator}")

    # Use normalization
    if config["normalize"]:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=config["normalize"]["norm_obs"],
            norm_reward=config["normalize"]["norm_reward"],
        )

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=f"./save/{args.env}",
        log_path=f"./logs/{args.env}",
        eval_freq=config["eval_freq"],
        deterministic=True,
        render=False,
    )

    # Set the optimizer class
    config["policy_kwargs"]["optimizer_class"] = torch.optim.AdamW

    model = PPO(
        policy=config["policy"],
        env=vec_env,
        n_steps=config["n_steps"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        n_epochs=config["n_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"], # cosine_schedule(config["learning_rate"]),
        clip_range=config["clip_range"],
        vf_coef=config["vf_coef"],
        ent_coef=config["ent_coef"],
        normalize_advantage=config["normalize_advantage"],
        max_grad_norm=config["max_grad_norm"],
        policy_kwargs=config["policy_kwargs"],
        verbose=0,
        tensorboard_log=f"./logs/{args.env}",
    )
    model.learn(
        total_timesteps=config["n_timesteps"],
        callback=[eval_callback, WandbCallback(verbose=2)],
        progress_bar=True,
    )

    run.finish()
