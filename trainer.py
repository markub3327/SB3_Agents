#!/usr/bin/env python
# coding: utf-8

# In[1]:


import stable_retro as retro
import ale_py
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame, MaxAndSkipEnv
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from wandb.integration.sb3 import WandbCallback
from gymnasium.wrappers import TimeLimit
import gymnasium
import numpy as np


# In[ ]:


class SimpleLinearSchedule:
    """
    Linear learning rate schedule (from initial value to zero),
    simpler than sb3 LinearSchedule.

    :param initial_value: (float or str) The initial value for the schedule
    """

    def __init__(self, initial_value: float | str) -> None:
        # Force conversion to float
        self.initial_value = float(initial_value)

    def __call__(self, progress_remaining: float) -> float:
        return progress_remaining * self.initial_value

    def __repr__(self) -> str:
        return f"SimpleLinearSchedule(initial_value={self.initial_value})"


def linear_schedule(initial_value: float | str) -> SimpleLinearSchedule:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: A `SimpleLinearSchedule` object
    """
    return SimpleLinearSchedule(initial_value)


# In[2]:

def make_retro_env():
        """
        Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
        """
        env = retro.make(env_name, retro.State.DEFAULT, render_mode=None)
        env = TimeLimit(env, max_episode_steps=8192)
        env = Monitor(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env, width=96, height=96)
        env = ClipRewardEnv(env)
        return env

def process_main(env_name):
    run = wandb.init(
        project="ppo-sb3",
        config={"env_name": env_name},
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    # For Atari console
    # vec_env = make_atari_env(env_name, n_envs=16, seed=1234)

    # For stable-retro consoles
    vec_env = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_retro_env] * 16), n_stack=4))

    # Use deterministic actions for evaluation
    # early_stopping = StopTrainingOnNoModelImprovement(
    #     max_no_improvement_evals=20,
    #     min_evals=10,
    #     verbose=1,
    # )
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=f"./save/{env_name}",
        log_path=f"./logs/{env_name}",
        eval_freq=10000,
        deterministic=True,
        render=False,
        # callback_after_eval=early_stopping,
    )

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        n_steps=8192,
        gamma=0.99,
        gae_lambda=0.70,
        n_epochs=4,
        batch_size=8192,
        learning_rate=linear_schedule(2.5e-4),
        clip_range=0.10,
        vf_coef=0.5,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log=f"./logs/{env_name}",
    )
    model.learn(total_timesteps=100_000_000, callback=[eval_callback, WandbCallback(verbose=2)], progress_bar=True)

    run.finish()


env_names = [
    "BoxingNoFrameskip-v4",
    "DemonAttackNoFrameskip-v4",
    "FishingDerbyNoFrameskip-v4",
    "FreewayNoFrameskip-v4",
    "GopherNoFrameskip-v4",
    "KrullNoFrameskip-v4",
    "KungFuMasterNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "AssaultNoFrameskip-v4",
    "VideoPinballNoFrameskip-v4",
    "StarGunnerNoFrameskip-v4",
    "CrazyClimberNoFrameskip-v4",
    "AtlantisNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "NameThisGameNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
    "TutankhamNoFrameskip-v4",
    "UpNDownNoFrameskip-v4",
    "BankHeistNoFrameskip-v4",
    "DoubleDunkNoFrameskip-v4",
    "JamesbondNoFrameskip-v4",
    "KangarooNoFrameskip-v4",
    "QbertNoFrameskip-v4",

    "SonicTheHedgehog2-Genesis-v0",
    "SonicTheHedgehog3-Genesis-v0",
    "SonicAndKnuckles3-Genesis-v0",

    "SuperMarioBros3-Nes-v0",
]


if __name__ == "__main__":
    for env_name in env_names:
        process_main(env_name)
