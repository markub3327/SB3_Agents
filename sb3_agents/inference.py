#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import ale_py
import cv2
import gymnasium
import mars_explorer
import numpy as np
import pandas as pd
import stable_retro as retro
from scipy.special import softmax
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
)
from tqdm import tqdm
from utils import ImageFilterForQueue, load_hyperparams, rollout, ids_action_vocab

from datasets import Dataset, Features, Image, Sequence, Value

# Use a dummy audio driver
os.environ["SDL_AUDIODRIVER"] = "dummy"

gymnasium.register_envs(ale_py)
gymnasium.register_envs(mars_explorer)

# Optimized game list
env_names = [
    "AssaultNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "PhoenixNoFrameskip-v4",
    "GopherNoFrameskip-v4",
    "KungFuMasterNoFrameskip-v4",

    "LunarLander-v3",
    "CartPole-v1",
]


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
# env_names = [
#     ### Platformer games
#     "SonicTheHedgehog2-Genesis-v0",
#     "SonicTheHedgehog3-Genesis-v0",
#     "SonicAndKnuckles3-Genesis-v0",
#     "SuperMarioBros3-Nes-v0",
#     "Ristar-Genesis-v0",
#     "RocketKnightAdventures-Genesis-v0",
#     "CastleOfIllusion-Genesis-v0",
#     "QuackShot-Genesis-v0",
#     "Vectorman2-Genesis-v0",
#     "KidChameleon-Genesis-v0",
#     "CoolSpot-Genesis-v0",
#     "GreendogTheBeachedSurferDude-Genesis-v0",
#      "KirbysAdventure-Nes-v0",
#     "MegaMan2-Nes-v0",
#    "AdventureIsland3-Nes-v0",
#    "FelixTheCat-Nes-v0",
#     "LittleMermaid-Nes-v0",
#    "BuckyOHare-Nes-v0",
#     "KidIcarus-Nes-v0",
#     "Shatterhand-Nes-v0",
#     "RockinKats-Nes-v0",
#     "ViceProjectDoom-Nes-v0",
#     "BubsyII-Snes-v0",
#    "ActRaiser2-Snes-v0",
#    "Plok-Snes-v0",
#     ### Sport games
#     "SuperHangOn-Genesis-v0",
#     "NHL94-Genesis-v0",
#     "F1-Genesis-v0",
#     "EuropeanClubSoccer-Genesis-v0",
#     ### Arcade shooters
#     "BioHazardBattle-Genesis-v0"
#     "MUSHA-Genesis-v0"
#     "Truxton-Genesis-v0"
#     "GrindStormer-Genesis-v0"
#     "Hellfire-Genesis-v0"
#     "Gaiares-Genesis-v0"
#     "ElementalMaster-Genesis-v0"
#     "ZeroWing-Genesis-v0"
#     "Viewpoint-Genesis-v0"
#     "SteelEmpire-Genesis-v0"
#     "GradiusII-Nes-v0"
#     "LifeForce-Nes-v0"
#     "Zanac-Nes-v0"
#     "GunNac-Nes-v0"
#     "TwinBee-Nes-v0"
#     "Parodius-Nes-v0"
#     "TerraCresta-Nes-v0"
#     "BuraiFighter-Nes-v0"
#     "DragonSpiritTheNewLegend-Nes-v0"
#     "XeviousTheAvenger-Nes-v0"
#     "Jackal-Nes-v0"
#     "HeavyBarrel-Nes-v0"
#     "GuerrillaWar-Nes-v0"
#     "POWPrisonersOfWar-Nes-v0"
#     "SuperC-Nes-v0"
#     "AeroFighters-Snes-v0"
#     ### Action games
#     "StreetsOfRage3-Genesis-v0",
#     "GoldenAxeIII-Genesis-v0",
#     "TeenageMutantNinjaTurtlesTheHyperstoneHeist-Genesis-v0",
#     "DoubleDragonIITheRevenge-Nes-v0",
#     "TeenageMutantNinjaTurtlesIIITheManhattanProject-Nes-v0",
#     "FinalFight3-Snes-v0",
#     ### Puzzle / Classic games
#     "MsPacMan-Genesis-v0"
#     "PacMania-Genesis-v0"
#     "BalloonFight-Nes-v0"
#     "DonkeyKong-Nes-v0"
#     "BubbleBobble-Nes-v0"
#     "SnowBrothers-Nes-v0"
#     "Arkanoid-Nes-v0"
#     "Popeye-Nes-v0"
#     "BoulderDash-GameBoy-v0"
#     "GradiusTheInterstellarAssault-GameBoy-v0"
#     "BlockKuzushiGB-GameBoy-v0"
#     "Cameltry-Snes-v0"
#     "PacInTime-Snes-v0"
# ]

# Full Atari version
# env_names = [
#     "AssaultNoFrameskip-v4",
#     "AtlantisNoFrameskip-v4",
#     "BankHeistNoFrameskip-v4",
#     "BoxingNoFrameskip-v4",
#     "BreakoutNoFrameskip-v4",
#     "CrazyClimberNoFrameskip-v4",
#    "DefenderNoFrameskip-v4",
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
#    "PhoenixNoFrameskip-v4",
#     "PongNoFrameskip-v4",
#     "QbertNoFrameskip-v4",
#     "RoadRunnerNoFrameskip-v4",
#     "StarGunnerNoFrameskip-v4",
#     "TutankhamNoFrameskip-v4",
#     "UpNDownNoFrameskip-v4",
#     "VideoPinballNoFrameskip-v4",

# "ALE/Blackjack-v5",
# "ALE/VideoChess-v5",
# "ALE/Turmoil-v5",
# "ALE/Trondead-v5",
# "ALE/TicTacToe3D-v5",
# "ALE/Tetris-v5",
# "ALE/Surround-v5",
# "ALE/Superman-v5",
# "ALE/SpaceWar-v5",
# "ALE/Othello-v5",
# "ALE/MrDo-v5",
# "ALE/MiniatureGolf-v5",
# "ALE/LostLuggage-v5",
# "ALE/LaserGates-v5",
# "ALE/KingKong-v5",
# "ALE/KeystoneKapers-v5",
# "ALE/Kaboom-v5",
# "ALE/Hangman-v5",
# "ALE/Galaxian-v5",
# "ALE/Frogger-v5",
# "ALE/DonkeyKong-v5",
# "ALE/Casino-v5",
# "ALE/BasicMath-v5",
# ]


def make_retro_env(env_name):
    def _init():
        env = retro.make(env_name, retro.State.DEFAULT, render_mode="rgb_array")
        env = Monitor(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env, width=96, height=96)
        return env

    return _init


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on saved PPO agents and generate datasets/videos."
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments to run (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment creation (default: 42).",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=8192,
        help="Maximum length of a rollout episode (default: 8192).",
    )
    parser.add_argument(
        "--with-random",
        action="store_true",
        help="Include a random agent that selects actions uniformly at random",
    )
    parser.add_argument(
        "--save-to-disk",
        action="store_true",
        help="Save generated trajectories to ./dataset as compressed pickle files (.pkl.gz).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Record and save a video for the best-scoring environment instance to ./videos.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="Fallback number of stacked frames when config does not define frame_stack.",
    )
    args = parser.parse_args()

    if args.save_to_disk:
        # Load the Vision model
        img_filter = ImageFilterForQueue()
    else:
        img_filter = None

    results_agent = {}
    for env_name in tqdm(env_names):
        print(f"Generating the dataset for {env_name} environment.")

        # Stable Retro
        if "-Genesis" in env_name or "-Nes" in env_name or "-Snes" in env_name:
            # Load PPO configuration
            config = load_hyperparams("retro")
            # Create environment
            vec_env = VecTransposeImage(
                VecFrameStack(
                    SubprocVecEnv([make_retro_env(env_name)] * args.n_envs),
                    n_stack=config["frame_stack"],
                )
            )
        # Atari 2600
        elif "NoFrameskip" in env_name or "ALE" in env_name:
            # Load PPO configuration
            config = load_hyperparams("ale")
            # Create environment
            vec_env = make_atari_env(
                env_name,
                n_envs=args.n_envs,
                seed=args.seed,
                wrapper_kwargs={"clip_reward": False},
            )
            # Frame-stacking with 4 frames
            vec_env = VecFrameStack(vec_env, n_stack=config["frame_stack"])
            vec_env = VecTransposeImage(vec_env)
        # Classic
        else:
            # Load PPO configuration
            config = load_hyperparams(env_name)
            # Create environment
            vec_env = make_vec_env(env_name, n_envs=args.n_envs, seed=args.seed)
            if config["policy"] == "CnnPolicy":
                vec_env = VecTransposeImage(vec_env)

        # Use normalization
        if config["normalize"]:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=config["normalize"]["norm_obs"],
                norm_reward=config["normalize"]["norm_reward"],
            )

        # Store env name on vec_env for later use (logging/saving).
        setattr(vec_env, "env_name", env_name)

        # Load pre-trained model
        model = (
            PPO.load(
                f"./save/{env_name}/best_model.zip",
                env=vec_env,
                custom_objects={"learning_rate": lambda _: 0.0, "_last_obs": None},
            )
            if not args.with_random
            else None
        )

        (
            obs,
            states,
            actions,
            actions_logits,
            rewards,
            scores,
            terminated,
            truncated,
            started,
            lives,
            imgs_embed,
            steps,
        ) = rollout(
            vec_env,
            model,
            episode_length=args.episode_length,
            n_stack=config["frame_stack"] if "frame_stack" in config else args.window_size,
            img_embed_model=img_filter,
            random=args.with_random,
        )

        if args.save_to_disk:
            def dataset_generator(shards):
                for shard in shards:
                    env_id = shard // args.episode_length
                    idx = shard % args.episode_length
                    print(f"Generating dataset for shard {shard}, that repsersent env {env_id} at timestep {idx}")
                    confidence = (
                        softmax(actions_logits[idx][env_id], axis=-1)[
                            ids_action_vocab[env_name][actions[idx][env_id]]
                        ]
                        * 100.0
                    )

                    if "CartPole" in env_name:
                        state = f"State description: Cart Position = {obs[idx][env_id][0]:.3f}; Cart Velocity = {obs[idx][env_id][1]:.3f}; Pole Angle = {obs[idx][env_id][2]:.3f}; Pole Angular Velocity = {obs[idx][env_id][3]:.3f}."
                    elif "LunarLander" in env_name:
                        state = f"State: horizontal position = {obs[idx][env_id][0]:.3f}; vertical position = {obs[idx][env_id][1]:.3f}; horizontal speed = {obs[idx][env_id][2]:.3f}; vertical speed = {obs[idx][env_id][3]:.3f}; tilt angle = {obs[idx][env_id][4]:.3f}; rotation speed = {obs[idx][env_id][5]:.3f}; left leg touching ground = {obs[idx][env_id][6]:.3f}; right leg touching ground = {obs[idx][env_id][7]:.3f}."
                    else:
                        state = None

                    example = {
                        "messages": {
                            "name": env_name,
                            "state": state,
                            "action": actions[idx][env_id],
                            "reward": rewards[idx][env_id],
                            "score": scores[idx][env_id],
                            "lives": lives[idx][env_id],
                            "terminated": terminated[idx][env_id],
                            "truncated": truncated[idx][env_id],
                            "started": started[idx][env_id],
                            "img_embed": imgs_embed[idx][env_id],
                            "confidence": confidence,
                            "reasoning": None,
                            "step": steps[idx][env_id],
                        },
                        "images": states[idx][env_id],
                    }
                    # print(example)

                    yield example

            # load datasets from folder
            shards = list(range(args.n_envs * states.shape[0]))
            cpus = os.cpu_count()
            dataset = Dataset.from_generator(
                dataset_generator,
                features=Features(
                    {
                        "messages": {
                            "name": Value("string"),
                            "state": Value("string"),
                            "action": Value("string"),
                            "reward": Value("float32"),
                            "score": Value("float32"),
                            "lives": Value("int64"),
                            "terminated": Value("bool"),
                            "truncated": Value("bool"),
                            "started": Value("bool"),
                            "confidence": Value("float32"),
                            "reasoning": Value("string"),
                            "step": Value("int64"),
                            "img_embed": Sequence(Sequence(Value("float32"))),
                        },
                        "images": Sequence(Image()),
                    }
                ),
                num_proc=cpus,
                gen_kwargs={"shards": shards},
            )
            print("Total samples:", len(dataset))

            # Save the dataset
            ds_path = "/mnt/data/home/makuke637/SB3_Agents/dataset"
            os.makedirs(ds_path, exist_ok=True)
            dataset.save_to_disk(
                os.path.join(ds_path, f"{env_name}"),
                num_proc=cpus,
            )

        # Shuffle the dataset once before saving
        dataset = dataset.shuffle(seed=42)
        print(f"Dataset shuffled with seed 42")

        # Store the results
        completed = np.logical_or(terminated, truncated)
        max_scores = np.nanmax(np.where(completed, scores, np.nan), axis=0)
        results_agent[env_name] = max_scores.tolist()

        # Recorder
        if args.save_video:
            best_idx = np.argmax(results_agent[env_name])
            height, width, channels = states[0, best_idx, -1].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            env_name = env_name.replace("ALE/", "")
            os.makedirs("./videos/", exist_ok=True)
            video = cv2.VideoWriter(
                f"./videos/{env_name}.mp4", fourcc, 60, (width, height)
            )
            for i in range(states.shape[0]):
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(states[i, best_idx, -1], cv2.COLOR_RGB2BGR)

                # Add text to the frame
                cv2.putText(
                    bgr_frame,
                    f"Action: {actions[i, best_idx]}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (112, 128, 144),  # Color (BGR)
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr_frame,
                    f"Reward: {rewards[i, best_idx]}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (112, 128, 144),  # Color (BGR)
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr_frame,
                    f"Score: {scores[i, best_idx]}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (112, 128, 144),  # Color (BGR)
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr_frame,
                    f"Lives: {lives[i, best_idx]}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (112, 128, 144),  # Color (BGR)
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr_frame,
                    f"Started: {started[i, best_idx]}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (112, 128, 144),  # Color (BGR)
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr_frame,
                    f"Terminated: {terminated[i, best_idx]}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (112, 128, 144),  # Color (BGR)
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr_frame,
                    f"Truncated: {truncated[i, best_idx]}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (112, 128, 144),  # Color (BGR)
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr_frame,
                    f"Step: {steps[i, best_idx]}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (112, 128, 144),  # Color (BGR)
                    1,
                    cv2.LINE_AA,
                )

                video.write(bgr_frame)
            video.release()
            print("Video recorded.")

        # Close envs
        vec_env.close()

    # Save to CSV file
    print(results_agent)
    df = pd.DataFrame(results_agent).T
    df.to_csv("results.csv")
