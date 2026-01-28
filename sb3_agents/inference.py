#!/usr/bin/env python
# coding: utf-8

# In[10]:

import gzip
import os
import pickle
import shutil

import ale_py
import cv2
import gymnasium
import numpy as np
import pandas as pd
import stable_retro as retro
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (SubprocVecEnv, VecFrameStack,
                                              VecTransposeImage)
from utils import rollout

# The best policies from mini PPO agent compared to [https://slm-lab.gitbook.io/slm-lab/benchmark-results/atari-benchmark]

# Classic
# env_names = [
# "LunarLander-v3",
# "Taxi-v3",
# "FrozenLake-v1",
# "Acrobot-v1",
# "CartPole-v1",
# "MountainCar-v0",
# ]

# Stable-retro
env_names = [
    ### Platformer games
    "SonicTheHedgehog2-Genesis-v0"
    "SonicTheHedgehog3-Genesis-v0"
    "SonicAndKnuckles3-Genesis-v0"
    "SuperMarioBros3-Nes-v0"
    "Ristar-Genesis-v0"
    "RocketKnightAdventures-Genesis-v0"
    "CastleOfIllusion-Genesis-v0"
    "QuackShot-Genesis-v0"
    "Vectorman2-Genesis-v0"
    "KidChameleon-Genesis-v0"
    "CoolSpot-Genesis-v0"
    "GreendogTheBeachedSurferDude-Genesis-v0"
    "KirbysAdventure-Nes-v0"
    "MegaMan2-Nes-v0"
    "AdventureIsland3-Nes-v0"
    "FelixTheCat-Nes-v0"
    "LittleMermaid-Nes-v0"
    "BuckyOHare-Nes-v0"
    "KidIcarus-Nes-v0"
    "Shatterhand-Nes-v0"
    "RockinKats-Nes-v0"
    "ViceProjectDoom-Nes-v0"
    "BubsyII-Snes-v0"
    "ActRaiser2-Snes-v0"
    "Plok-Snes-v0"
    ### Sport games
    "SuperHangOn-Genesis-v0"
    "NHL94-Genesis-v0"
    "F1-Genesis-v0"
    "EuropeanClubSoccer-Genesis-v0"
    ### Arcade shooters
    "BioHazardBattle-Genesis-v0"
    "MUSHA-Genesis-v0"
    "Truxton-Genesis-v0"
    "GrindStormer-Genesis-v0"
    "Hellfire-Genesis-v0"
    "Gaiares-Genesis-v0"
    "ElementalMaster-Genesis-v0"
    "ZeroWing-Genesis-v0"
    "Viewpoint-Genesis-v0"
    "SteelEmpire-Genesis-v0"
    "GradiusII-Nes-v0"
    "LifeForce-Nes-v0"
    "Zanac-Nes-v0"
    "GunNac-Nes-v0"
    "TwinBee-Nes-v0"
    "Parodius-Nes-v0"
    "TerraCresta-Nes-v0"
    "BuraiFighter-Nes-v0"
    "DragonSpiritTheNewLegend-Nes-v0"
    "XeviousTheAvenger-Nes-v0"
    "Jackal-Nes-v0"
    "HeavyBarrel-Nes-v0"
    "GuerrillaWar-Nes-v0"
    "POWPrisonersOfWar-Nes-v0"
    "SuperC-Nes-v0"
    "AeroFighters-Snes-v0"
    ### Action games
    "StreetsOfRage3-Genesis-v0"
    "GoldenAxeIII-Genesis-v0"
    "TeenageMutantNinjaTurtlesTheHyperstoneHeist-Genesis-v0"
    "DoubleDragonIITheRevenge-Nes-v0"
    "TeenageMutantNinjaTurtlesIIITheManhattanProject-Nes-v0"
    "FinalFight3-Snes-v0"
    ### Puzzle / Classic games
    "MsPacMan-Genesis-v0"
    "PacMania-Genesis-v0"
    "BalloonFight-Nes-v0"
    "DonkeyKong-Nes-v0"
    "BubbleBobble-Nes-v0"
    "SnowBrothers-Nes-v0"
    "Arkanoid-Nes-v0"
    "Popeye-Nes-v0"
    "BoulderDash-GameBoy-v0"
    "GradiusTheInterstellarAssault-GameBoy-v0"
    "BlockKuzushiGB-GameBoy-v0"
    "Cameltry-Snes-v0"
    "PacInTime-Snes-v0"
]

# Optimalized Atari version
env_names = [
    "GopherNoFrameskip-v4",
    "NameThisGameNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "AssaultNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "FreewayNoFrameskip-v4",
    "StarGunnerNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "BoxingNoFrameskip-v4",
    "FishingDerbyNoFrameskip-v4",
]


# Full Atari version
env_names = [
#     "AssaultNoFrameskip-v4",
#     "AtlantisNoFrameskip-v4",
#     "BankHeistNoFrameskip-v4",
#     "BoxingNoFrameskip-v4",
#     "BreakoutNoFrameskip-v4",
#     "CrazyClimberNoFrameskip-v4",
#     "DemonAttackNoFrameskip-v4",
#     "DoubleDunkNoFrameskip-v4",
#     "EnduroNoFrameskip-v4",
#     "FishingDerbyNoFrameskip-v4",
#     "FreewayNoFrameskip-v4",
#     "GopherNoFrameskip-v4",
#     "JamesbondNoFrameskip-v4",
#     "KangarooNoFrameskip-v4",
#     "KrullNoFrameskip-v4",
#     "KungFuMasterNoFrameskip-v4",
#     "NameThisGameNoFrameskip-v4",
#     "PongNoFrameskip-v4",
#     "QbertNoFrameskip-v4",
#     "RoadRunnerNoFrameskip-v4",
#     "StarGunnerNoFrameskip-v4",
#     "TutankhamNoFrameskip-v4",
#     "UpNDownNoFrameskip-v4",
#     "VideoPinballNoFrameskip-v4",

    "ALE/Blackjack-v5",
    "ALE/VideoCube-v5",
    "ALE/VideoChess-v5",
    "ALE/VideoCheckers-v5",
    "ALE/Turmoil-v5",
    "ALE/Trondead-v5",
    "ALE/TicTacToe3D-v5",
    "ALE/Tetris-v5",
    "ALE/Surround-v5",
    "ALE/Superman-v5",
    "ALE/SpaceWar-v5",
    "ALE/Othello-v5",
    "ALE/MrDo-v5",
    "ALE/MiniatureGolf-v5",
    "ALE/LostLuggage-v5",
    "ALE/LaserGates-v5",
    "ALE/KingKong-v5",
    "ALE/KeystoneKapers-v5",
    "ALE/Kaboom-v5",
    "ALE/Hangman-v5",
    "ALE/Galaxian-v5",
    "ALE/Frogger-v5",
    "ALE/DonkeyKong-v5",
    "ALE/Casino-v5",
    "ALE/BasicMath-v5",
]


def make_retro_env(env_name):
    def _init():
        env = retro.make(env_name, retro.State.DEFAULT, render_mode="rgb_array")
        env = TimeLimit(env, max_episode_steps=8192)
        env = Monitor(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env, width=96, height=96)
        return env

    return _init


if __name__ == "__main__":
    n_envs, n_stack, mySeed, results = 64, 4, 1234, {}
    for env_name in env_names:
        print(f"Generating the dataset for {env_name} environment.")

        # Stable Retro
        if "-Genesis" in env_name or "-Nes" in env_name or "-Snes" in env_name:
            vec_env = VecTransposeImage(
                VecFrameStack(
                    SubprocVecEnv([make_retro_env(env_name)] * n_envs), n_stack=n_stack
                )
            )
        # Atari 2600
        elif "NoFrameskip" in env_name or "ALE" in env_name:
            # There already exists an environment generator
            # that will make and wrap atari environments correctly.
            # Here we are also multi-worker training (n_envs=4 => 4 environments)
            vec_env = make_atari_env(
                env_name,
                n_envs=n_envs,
                seed=mySeed,
                wrapper_kwargs={"clip_reward": False},
            )
            # Frame-stacking with 4 frames
            vec_env = VecFrameStack(vec_env, n_stack=n_stack)
            vec_env = VecTransposeImage(vec_env)
        # Classic
        else:
            vec_env = make_vec_env(env_name, n_envs=n_envs, seed=mySeed)

        # Store env name on vec_env for later use (logging/saving).
        setattr(vec_env, "env_name", env_name)

        # Load pre-trained model from file
        model = PPO.load(
            f"./save/{env_name}/best_model.zip",
            env=vec_env,
            custom_objects={"learning_rate": lambda _: 0.0},
        )

        (states, actions, actions_logits, rewards, terminated, truncated, lives) = (
            rollout(vec_env, model, random=False)
        )

        for sample_id in range(n_envs):
            # Save the dataset
            ds_path = f"./dataset/sample_{sample_id}"
            os.makedirs(ds_path, exist_ok=True)

            n = states.shape[0]
            window_size = (
                n if n < n_stack else np.random.randint(n_stack, min(n, 128) + 1)
            )
            print("window_size:", window_size)
            data = {
                "name": env_name,
                "states": np.array_split(
                    states[:, sample_id], states.shape[0] // window_size
                ),
                "actions": np.array_split(
                    actions[:, sample_id], actions.shape[0] // window_size
                ),
                "actions_logits": np.array_split(
                    actions_logits[:, sample_id], actions_logits.shape[0] // window_size
                ),
                "rewards": np.array_split(
                    rewards[:, sample_id], rewards.shape[0] // window_size
                ),
                "terminated": np.array_split(
                    terminated[:, sample_id], terminated.shape[0] // window_size
                ),
                "truncated": np.array_split(
                    truncated[:, sample_id], truncated.shape[0] // window_size
                ),
                "lives": np.array_split(
                    lives[:, sample_id], lives.shape[0] // window_size
                ),
            }

            with gzip.open(os.path.join(ds_path, f"{env_name}.pkl.gz"), "wb") as f:
                pickle.dump(data, f)

        print(f"Generated was {states.shape[0] * n_envs} samples.")

        # Store the results
        results[env_name] = np.max(rewards, axis=0)

        # Recorder
        best_idx = np.argmax(results[env_name])
        height, width, channels = states[0, best_idx].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        env_name = env_name.replace("ALE/", "")
        video = cv2.VideoWriter(
            f"../videos/{env_name}.mp4", fourcc, 60, (width, height)
        )
        for i in range(states.shape[0]):
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(states[i, best_idx], cv2.COLOR_RGB2BGR)
            video.write(bgr_frame)
        video.release()
        print("Video recorded.")

        # Close envs
        vec_env.close()

    # Save to CSV file
    print(results)
    df = pd.DataFrame(results)
    df.to_csv("results.csv")
