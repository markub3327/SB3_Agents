#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import cv2
import ale_py
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage, VecNormalize
from tqdm import tqdm
from utils import load_hyperparams, ImageFilterForQueue
from datasets import Dataset, Features, Value, Image, Sequence
from gymnasium.spaces import Box
from vocab import ids_action_vocab
from collections import deque


# Use a dummy audio driver
os.environ["SDL_AUDIODRIVER"] = "dummy"

gymnasium.register_envs(ale_py)

# Full Atari version
env_names = (
    "AssaultNoFrameskip-v4",
    "AtlantisNoFrameskip-v4",
    "BankHeistNoFrameskip-v4",
    "BoxingNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "CrazyClimberNoFrameskip-v4",
    "DefenderNoFrameskip-v4",
    "DemonAttackNoFrameskip-v4",
    "DoubleDunkNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "FishingDerbyNoFrameskip-v4",
    "FreewayNoFrameskip-v4",
    "GopherNoFrameskip-v4",
    "JamesbondNoFrameskip-v4",
    "KangarooNoFrameskip-v4",
    "KrullNoFrameskip-v4",
    "KungFuMasterNoFrameskip-v4",
    "PhoenixNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
    "StarGunnerNoFrameskip-v4",
    "TutankhamNoFrameskip-v4",
    "UpNDownNoFrameskip-v4",
    "VideoPinballNoFrameskip-v4"
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
        default=16,
        help="Number of parallel environments to run (default: 16).",
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
            os.path.join("./videos/", f"{env_name}.mp4".replace("/", "_")),
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (400, 400),
        )

        # Stable Retro
        if "-Genesis" in env_name or "-Nes" in env_name or "-Snes" in env_name:
            # Load PPO configuration
            config = load_hyperparams("retro")
            # Create environment
            vec_env = SubprocVecEnv([make_retro_env(args.env)] * args.n_envs)
            vec_env = VecFrameStack(vec_env, n_stack=config["frame_stack"])
            vec_env = VecTransposeImage(vec_env)
            vec_env.action_space.seed(args.seed)
            vec_env.seed(args.seed)
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
                custom_objects={"learning_rate": lambda _: 0.0, "_last_obs": None},
            )
            if not args.with_random
            else None
        )

        # Initialize lists
        state_list = []
        action_list = []
        reward_list = []
        value_list = []
        done_list = []
        started_list = []
        imgs_embed_list = []

        # Start environment
        started_list.append(np.ones([vec_env.num_envs], dtype=np.bool))
        obs = vec_env.reset()

        # State
        rendered_img = vec_env.env_method("render")
        state_history = deque(
            iterable=[rendered_img],
            maxlen=config["frame_stack"]
        )

        for i in range(args.episode_length):
            print(f"Step {i}")

            if img_filter:
                print("rendered_img", len(state_history))
                # img_batch = np.stack(rendered_img, axis=0)
                img_batch = list(state_history)
                img_embed = img_filter.get_embedding(img_batch)
                print("img_embed shape", img_embed.shape)
                img_embed = np.stack(np.split(img_embed, vec_env.num_envs, axis=0), axis=0)
                print("img_embed shape", img_embed.shape)
                imgs_embed_list.append(img_embed)

            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            values = model.policy.predict_values(obs_tensor).detach().cpu().numpy()

            # Make step
            obs, reward, done, _ = vec_env.step(action)

            # Information
            state_list.append(np.asarray(state_history))
            action_name = [ids_action_vocab[env_name].inverse[a] for a in action]
            action_list.append(np.asarray(action_name))
            reward_list.append(reward)
            value_list.append(values)
            done_list.append(done)
            print("action", action_name, "reward", reward, "done", done)

            # Update image
            rendered_img = vec_env.env_method("render")
            state_history.append(rendered_img)

            # Update started
            started_list.append(done)

        # print(state_list)
        # print(action_list)
        # print(reward_list)
        # print(value_list)
        # print(done_list)
        # print(started_list)
        for k in range(args.episode_length):
            frame = cv2.resize(state_list[k][-1][0], (400, 400))
            # Add text to the frame
            cv2.putText(
                frame,
                f"Action: {action_list[k][0]}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (128, 128, 144),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Reward: {reward_list[k][0]}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (128, 128, 144),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Terminated: {done_list[k][0]}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (128, 128, 144),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Started: {started_list[k][0]}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (128, 128, 144),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Value: {value_list[k][0]}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (128, 128, 144),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Env ID: {k}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (128, 128, 144),
                1,
                cv2.LINE_AA,
            )
            video_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_out.release()

        # Close envs
        vec_env.close()

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
                                "value": value_list[shard][i],
                                "done": done_list[shard][i],
                                "started": started_list[shard][i],
                                "reasoning": None,
                                "img_embed": imgs_embed_list[shard][i]
                            },
                            "images": state_list[shard][:, i]
                        }

                        yield example

            # load datasets from folder
            shards = list(range(args.episode_length))
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
                            "value": Value("float32"),
                            "done": Value("bool"),
                            "started": Value("bool"),
                            "reasoning": Value("string"),
                            "img_embed": Sequence(Sequence(Value("float32")))
                        },
                        "images": Sequence(Image()),
                    }
                ),
                num_proc=cpus,
                gen_kwargs={"shards": shards},
            )
            print("Total samples:", len(dataset))

            # Save the dataset
            ds_path = "/mnt/home/makuke637/SB3_Agents/dataset"
            os.makedirs(ds_path, exist_ok=True)
            dataset.save_to_disk(
                os.path.join(ds_path, f"{env_name}"),
                num_proc=cpus,
            )