#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import cv2
import ale_py
import gymnasium
import mars_explorer
import numpy as np
import stable_retro as retro
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from tqdm import tqdm
from utils import ImageFilterForQueue, load_hyperparams
from datasets import Dataset, Features, Value, Image, Sequence
from gymnasium.spaces import Box
from vocab import ids_action_vocab


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
        default=4,
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

    args = parser.parse_args()

    if args.save_to_disk:
        # Load the Vision model
        img_filter = ImageFilterForQueue()
    else:
        img_filter = None

    for env_name in tqdm(env_names):
        print(f"Generating the dataset for {env_name} environment.")

        video_out = cv2.VideoWriter(
            os.path.join("./videos/", f"{env_name}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (400, 400),
        )

        # Stable Retro
        if "-Genesis" in env_name or "-Nes" in env_name or "-Snes" in env_name:
            # Load PPO configuration
            config = load_hyperparams("retro")
            # Create environment
            # Frame-stacking with 4 frames
            vec_env = SubprocVecEnv([make_retro_env(env_name)] * args.n_envs)
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
            vec_env = VecFrameStack(vec_env, n_stack=config["frame_stack"])
            vec_env = VecTransposeImage(vec_env)
        # Classic
        else:
            # Load PPO configuration
            config = load_hyperparams(env_name)
            # Create environment
            vec_env = make_vec_env(env_name, n_envs=args.n_envs, seed=args.seed)

        # Store env name on vec_env for later use (logging/saving).
        setattr(vec_env, "env_name", env_name)

        # Load pre-trained model
        model = (
            PPO.load(
                f"./save/{env_name}/best_model.zip",
                custom_objects={"learning_rate": lambda _: 0.0, "_last_obs": None},
            )
            if not args.with_random
            else None
        )

        state_list = []
        action_list = []
        reward_list = []
        score_list = []
        done_list = []
        started_list = []
        imgs_embed_list = []

        score = np.zeros((vec_env.num_envs,))
        started_list.extend(np.ones((vec_env.num_envs,), dtype=np.bool))
        obs, _ = vec_env.reset()

        rendered_img = vec_env.env_method("render")
        stacked_obs = StackedObservations(
            vec_env.num_envs,
            config["frame_stack"],
            Box(0, 255, rendered_img[0].shape, dtype=np.uint8),
        )
        rendered_img = stacked_obs.reset(
            np.asarray(rendered_img, dtype=np.uint8)
        )
        rendered_img = np.stack(np.split(rendered_img, config["frame_stack"], axis=-1), axis=1)

        for i in range(args.episode_length):
            action, _ = model.predict(obs, deterministic=True)
            state_list.extend(rendered_img)

            if img_filter:
                print("rendered_img", rendered_img.shape)
                img_batch = rendered_img.reshape(-1, rendered_img.shape[2], rendered_img.shape[3], rendered_img.shape[4])
                img_embed = img_filter.get_embedding(img_batch)
                print("img_embed shape", img_embed.shape)
                img_embed = np.stack(np.split(img_embed, vec_env.num_envs, axis=0), axis=0)
                print("img_embed shape", img_embed.shape)
                imgs_embed_list.extend(img_embed)

            obs, reward, done, info = vec_env.step(action)
            action = [[ids_action_vocab[env_name].inverse[a]] for a in action]
            action_list.extend(action)
            reward_list.extend(reward)
            done_list.extend(done)
            score_list.extend(score)
            score += reward
            print("action", action, "reward", reward, "score", score, "done", done, "info", info)

            print(rendered_img.shape)
            for k in range(vec_env.num_envs):
                for j, img in enumerate(rendered_img[k]):
                    frame = cv2.resize(img, (200, 200))
                    # Add text to the frame
                    cv2.putText(
                        frame,
                        f"Action: {action}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (112, 128, 144),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Reward: {reward}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (112, 128, 144),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Score: {score}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (112, 128, 144),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Terminated: {done}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (112, 128, 144),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Started: {started_list}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (112, 128, 144),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Frame ID: {j}",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (112, 128, 144),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Env ID: {k}",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (112, 128, 144),
                        1,
                        cv2.LINE_AA,
                    )
                    video_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            rendered_img = vec_env.env_method("render")
            rendered_img, _ = stacked_obs.update(
                np.asarray(rendered_img, dtype=np.uint8),
                done,
                # info
                ([{}] * vec_env.num_envs)
            )
            rendered_img = np.stack(np.split(rendered_img, config["frame_stack"], axis=-1), axis=1)
            print(rendered_img.shape)

            # Reset score counter
            finished = np.where(done)[0]
            if len(finished) > 0:
                print(f"Finished envs: {finished}")
                score[finished] = 0

            started_list.extend(done)

        # Close envs
        vec_env.close()
        video_out.release()

        print(len(state_list), len(action_list), len(reward_list), len(done_list), len(score_list), started_list)

        if args.save_to_disk:
            def dataset_generator(shards):
                for shard in shards:
                    print(f"Generating dataset for shard {shard}")

                    example = {
                        "messages": {
                            "name": env_name,
                            "state": None,
                            "action": action_list[shard],
                            "reward": reward_list[shard],
                            "score": score_list[shard],
                            "done": done_list[shard],
                            "started": started_list[shard],
                            "reasoning": None
                        },
                        "images": state_list[shard],
                    }
                    # print(example)

                    yield example

            # load datasets from folder
            shards = list(range(len(state_list)))
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
                            "done": Value("bool"),
                            "started": Value("bool"),
                            "reasoning": Value("string"),
                        },
                        "images": Sequence(Image()),
                    }
                ),
                num_proc=cpus,
                gen_kwargs={"shards": shards},
            )
            print("Total samples:", len(dataset))

            # Shuffle the dataset once before saving
            dataset = dataset.shuffle(seed=42)
            print(f"Dataset shuffled with seed 42")

            # Save the dataset
            ds_path = "/mnt/data/home/makuke637/SB3_Agents/dataset"
            os.makedirs(ds_path, exist_ok=True)
            dataset.save_to_disk(
                os.path.join(ds_path, f"{env_name}"),
                num_proc=cpus,
            )