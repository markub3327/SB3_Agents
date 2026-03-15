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
from utils import ImageFilterForQueue, load_hyperparams, random_splits, rollout

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
    "Acrobot-v1",
    "CartPole-v1",
    "MountainCar-v0",
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
        env = TimeLimit(env, max_episode_steps=8192)
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
        default=64,
        help="Number of parallel environments to run (default: 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment creation (default: 42).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="Max trajectory chunk length used when splitting rollouts into multiple samples. (default: 128).",
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
                custom_objects={"learning_rate": lambda _: 0.0},
            )
            if not args.with_random
            else None
        )

        (
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
        ) = rollout(
            vec_env,
            model,
            episode_length=args.episode_length,
            img_embed_model=img_filter,
            random=args.with_random,
        )

        if args.save_to_disk:

            def dataset_generator(shards):
                for env_id in shards:
                    print(f"Generating dataset for shard {env_id}")
                    s, a, a_logits, r, score, term, trunc, start, l, img_embed = (
                        random_splits(
                            states[:, env_id],
                            actions[:, env_id],
                            actions_logits[:, env_id],
                            rewards[:, env_id],
                            scores[:, env_id],
                            terminated[:, env_id],
                            truncated[:, env_id],
                            started[:, env_id],
                            lives[:, env_id],
                            imgs_embed[:, env_id],
                            min_size=(
                                config["frame_stack"] if "frame_stack" in config else 2
                            ),
                            max_size=args.window_size,
                        )
                    )

                    # Check number of splits
                    assert (
                        len(s)
                        == len(a)
                        == len(a_logits)
                        == len(r)
                        == len(score)
                        == len(term)
                        == len(trunc)
                        == len(start)
                        == len(l)
                        == len(img_embed)
                    ), (
                        f"Split length mismatch: s={len(s)}, a={len(a)}, "
                        f"a_logits={len(a_logits)}, r={len(r)}, score={len(score)}, term={len(term)}, "
                        f"trunc={len(trunc)}, start={len(start)}, l={len(l)}, img_embed={len(img_embed)}"
                    )

                    for idx in range(len(s)):
                        # Check minimal number frames for video
                        assert (
                            len(s[idx])
                            == len(a[idx])
                            == len(a_logits[idx])
                            == len(r[idx])
                            == len(score[idx])
                            == len(term[idx])
                            == len(trunc[idx])
                            == len(start[idx])
                            == len(l[idx])
                            == len(img_embed[idx])
                            > 1
                        ), (
                            f"Split length mismatch: s={len(s[idx])}, a={len(a[idx])}, "
                            f"a_logits={len(a_logits[idx])}, r={len(r[idx])}, score={len(score[idx])}, term={len(term[idx])}, "
                            f"trunc={len(trunc[idx])}, start={len(start[idx])}, l={len(l[idx])}, img_embed={len(img_embed[idx])}"
                        )

                        example = {
                            "messages": {
                                "name": env_name.replace("NoFrameskip-v4", ""),
                                "action": a[idx],
                                "action_logits": a_logits[idx],
                                "reward": r[idx],
                                "score": score[idx],
                                "terminated": term[idx],
                                "truncated": trunc[idx],
                                "started": start[idx],
                                "lives": l[idx],
                                "img_embed": img_embed[idx],
                            },
                            "images": s[idx],
                        }

                        yield example

            # load datasets from folder
            shards = list(range(args.n_envs))
            cpus = os.cpu_count()
            dataset = Dataset.from_generator(
                dataset_generator,
                features=Features(
                    {
                        "messages": {
                            "name": Value("string"),  # ok
                            "action": Sequence(Value("string")),  # ok
                            "action_logits": Sequence(Sequence(Value("float32"))),
                            "reward": Sequence(Value("float32")),  # ok
                            "score": Sequence(Value("float32")),  # ok
                            "lives": Sequence(Value("int64")),  # ok
                            "terminated": Sequence(Value("bool")),  # ok
                            "truncated": Sequence(Value("bool")),  # ok
                            "started": Sequence(Value("bool")),  # ok
                            "img_embed": Sequence(Sequence(Value("float32"))),  # ok
                        },
                        "images": Sequence(Image()),
                    }
                ),
                num_proc=cpus if cpus <= args.n_envs else args.n_envs,
                gen_kwargs={"shards": shards},
            )
            print("Total samples:", len(dataset))

            # Save the dataset
            ds_path = "/mnt/data/home/makuke637/SB3_Agents/dataset"
            os.makedirs(ds_path, exist_ok=True)
            dataset.save_to_disk(
                os.path.join(ds_path, f"{env_name}"),
                num_proc=min(args.n_envs, len(dataset)),
            )

        # Store the results
        results_agent[env_name] = np.max(
            scores[np.where(np.logical_or(terminated, truncated))[0]], axis=0
        ).tolist()

        # Recorder
        if args.save_video:
            best_idx = np.argmax(results_agent[env_name])
            height, width, channels = states[0, best_idx].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            env_name = env_name.replace("ALE/", "")
            os.makedirs("./videos/", exist_ok=True)
            video = cv2.VideoWriter(
                f"./videos/{env_name}.mp4", fourcc, 60, (width, height)
            )
            for i in range(states.shape[0]):
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(states[i, best_idx], cv2.COLOR_RGB2BGR)

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
                    f"Truncated: {truncated[i, best_idx]}",
                    (10, 90),
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
