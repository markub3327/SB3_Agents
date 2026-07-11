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
from utils import load_hyperparams
from datasets import Dataset, Features, Value, Image, Sequence
from gymnasium.spaces import Box
from vocab import ids_action_vocab
from collections import deque


# Use a dummy audio driver
os.environ["SDL_AUDIODRIVER"] = "dummy"

gymnasium.register_envs(ale_py)
gymnasium.register_envs(mars_explorer)

# Optimized game list
# env_names = [
#     # "AssaultNoFrameskip-v4",
#     # "BreakoutNoFrameskip-v4",
#     # "QbertNoFrameskip-v4",
#     # "PhoenixNoFrameskip-v4",
#     # "GopherNoFrameskip-v4",
#     # "KungFuMasterNoFrameskip-v4",
#     "LunarLander-v3",
#     "CartPole-v1",
# ]


# The best policies from mini PPO agent compared to [https://slm-lab.gitbook.io/slm-lab/benchmark-results/atari-benchmark]

# Classic
# env_names = [
#     "LunarLander-v3",
#     "Acrobot-v1",
#     "CartPole-v1",
#     "MountainCar-v0",
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
env_names = (
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

     "AdventureNoFrameskip-v4"
     "AirRaidNoFrameskip-v4"
     "AlienNoFrameskip-v4"
     "AmidarNoFrameskip-v4"
     "AsterixNoFrameskip-v4"
     "AsteroidsNoFrameskip-v4"
     "Atlantis2NoFrameskip-v4"
     "BackgammonNoFrameskip-v4"
     "BattleZoneNoFrameskip-v4"
     "BeamRiderNoFrameskip-v4"
     "BerzerkNoFrameskip-v4"
     "BowlingNoFrameskip-v4"
     "CarnivalNoFrameskip-v4"
     "CentipedeNoFrameskip-v4"
     "ChopperCommandNoFrameskip-v4"
     "CrossbowNoFrameskip-v4"
     "DarkchambersNoFrameskip-v4"
     "EarthworldNoFrameskip-v4"
     "ElevatorActionNoFrameskip-v4"
     "EntombedNoFrameskip-v4"
     "EtNoFrameskip-v4"
     "FlagCaptureNoFrameskip-v4"
     "FrostbiteNoFrameskip-v4"
     "GravitarNoFrameskip-v4"
     "HauntedHouseNoFrameskip-v4"
     "HeroNoFrameskip-v4"
     "HumanCannonballNoFrameskip-v4"
     "IceHockeyNoFrameskip-v4"
     "JourneyEscapeNoFrameskip-v4"
     "KlaxNoFrameskip-v4"
     "KoolaidNoFrameskip-v4"
     "MarioBrosNoFrameskip-v4"
     "MontezumaRevengeNoFrameskip-v4"
     "MsPacmanNoFrameskip-v4"
     "PacmanNoFrameskip-v4"
     "PitfallNoFrameskip-v4"
     "Pitfall2NoFrameskip-v4"
     "PooyanNoFrameskip-v4"
     "PrivateEyeNoFrameskip-v4"
     "RiverraidNoFrameskip-v4"
     "RobotankNoFrameskip-v4"
     "SeaquestNoFrameskip-v4"
     "SirLancelotNoFrameskip-v4"
     "SkiingNoFrameskip-v4"
     "SolarisNoFrameskip-v4"
     "SpaceInvadersNoFrameskip-v4"
     "TennisNoFrameskip-v4"
     "TimePilotNoFrameskip-v4"
     "VentureNoFrameskip-v4"
     "VideoCubeNoFrameskip-v4"
     "WizardOfWorNoFrameskip-v4"
     "WordZapperNoFrameskip-v4"
     "YarsRevengeNoFrameskip-v4"
     "ZaxxonNoFrameskip-v4"
)


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
        default=8,
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

    for env_name in tqdm(env_names):
        print(f"Generating the dataset for {env_name} environment.")

        video_out = cv2.VideoWriter(
            os.path.join("./videos/", f"{env_name}.mp4".replace("/", "_")),
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
        done_list = []
        started_list = []

        started_list.append(np.ones((vec_env.num_envs,), dtype=np.bool))
        obs, _ = vec_env.reset()

        # State
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

        # Action
        action_history = deque(iterable=([[None] * vec_env.num_envs] * config["frame_stack"]), maxlen=(config["frame_stack"] + 1))

        for i in range(args.episode_length):
            print(f"Step {i}")

            state_list.append(rendered_img)

            # Predict action
            action, _ = model.predict(obs, deterministic=True)

            # Make step
            obs, reward, done, info = vec_env.step(action)

            # Information
            action_name = [ids_action_vocab[env_name].inverse[a] for a in action]
            action_history.append(action_name)
            action_list.append(np.transpose(np.asarray(action_history), (1, 0)))
            reward_list.append(reward)
            done_list.append(done)
            print("action", action_history, "reward", reward, "done", done)

            # Update image
            rendered_img = vec_env.env_method("render")
            rendered_img, _ = stacked_obs.update(
                np.asarray(rendered_img, dtype=np.uint8),
                done,
                # info
                ([{}] * vec_env.num_envs)
            )
            rendered_img = np.stack(np.split(rendered_img, config["frame_stack"], axis=-1), axis=1)
            # print(rendered_img.shape)

            # Update started
            started_list.append(done)

        # Close envs
        vec_env.close()

        print(len(state_list), len(action_list), len(reward_list), len(done_list), len(started_list))

        for k in range(args.episode_length):
            # print(state_list[k][0].shape)
            frame = cv2.resize(state_list[k][0][0], (400, 400))
            # Add text to the frame
            cv2.putText(
                frame,
                f"Action: {action_list[k][0]}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (112, 128, 144),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Reward: {reward_list[k][0]}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (112, 128, 144),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Terminated: {done_list[k][0]}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (112, 128, 144),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Started: {started_list[k][0]}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (112, 128, 144),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Env ID: {k}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (112, 128, 144),
                1,
                cv2.LINE_AA,
            )
            video_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_out.release()

        if args.save_to_disk:
            def dataset_generator(shards):
                for shard in shards:
                    for i in range(args.n_envs):
                        print(f"Generating dataset for shard {shard}, env {i}")

                        example = {
                            "messages": {
                                "name": env_name,
                                "state": None,
                                "action": action_list[shard][i],
                                "reward": reward_list[shard][i],
                                "done": done_list[shard][i],
                                "started": started_list[shard][i],
                                "reasoning": None
                            },
                            "images": state_list[shard][i],
                        }

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
                            "action": Sequence(Value("string")),
                            "reward": Value("float32"),
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

            # Save the dataset
            ds_path = "/mnt/data/home/makuke637/SB3_Agents/dataset"
            os.makedirs(ds_path, exist_ok=True)
            dataset.save_to_disk(
                os.path.join(ds_path, f"{env_name}"),
                num_proc=cpus,
            )