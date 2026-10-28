import ale_py
import datetime
import gymnasium
import os
import json
import numpy as np
from stable_baselines3 import PPO
from utils import FrameFilterForQueue
from datasets import Dataset, Features, Value, Image, Version, Sequence
from collections import deque
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformReward


# Use a dummy audio driver
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Metadata
NUM_GAMES = 1

# Register environments
gymnasium.register_envs(ale_py)

### Atari 2600
env_names = (
    "AssaultNoFrameskip-v4",
    "BankHeistNoFrameskip-v4",
    "BoxingNoFrameskip-v4",
    "CrazyClimberNoFrameskip-v4",
    "DoubleDunkNoFrameskip-v4",
    "FishingDerbyNoFrameskip-v4",
    "FreewayNoFrameskip-v4",
    "GopherNoFrameskip-v4",
    "JamesbondNoFrameskip-v4",
    "KrullNoFrameskip-v4",
    "KungFuMasterNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
    "StarGunnerNoFrameskip-v4",
    "UpNDownNoFrameskip-v4",
)


def dataset_generator(shards, env_name, player_name):
    print(f"Generating the dataset for {env_name} environment.")

    # Initialize the frame filter
    img_filter = FrameFilterForQueue()

    # Load pre-trained model
    model = PPO.load(
        f"./save/{env_name}/best_model.zip",
        custom_objects={"learning_rate": lambda _: 0.0, "_last_obs": None},
    )

    # Init environment
    env = gymnasium.make(env_name, render_mode="rgb_array")
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        grayscale_obs=True,
    )
    env = FrameStackObservation(env, stack_size=4, padding_type="zero")
    env = TransformReward(env, lambda r: np.sign(float(r)))
    for shard in shards:
        # Initialize the frame queue
        frame_queue = deque(maxlen=8)

        # Init agent
        player = {
            "frames": [],
            "action": [],
            "reward": [],
            "lives": [],
            "status": [],
            "frames_similarity": [],
        }

        # Reset the environment for a new game
        observation, info = env.reset()
        print("info", info)

        # The initial frame
        frame = env.render()
        frame_queue.append(frame)

        # Run the game loop
        while True:
            # Render the current frame
            frame = env.render()
            frame_queue.append(frame)
            player['frames'].append(
                np.stack(frame_queue, axis=0)
            )

            # Get the embedding for the current frame queue
            print("rendered frames", len(frame_queue))
            frames_similarity = img_filter.get_similarity(frame_queue)
            print("frames_similarity shape", frames_similarity.shape)
            player['frames_similarity'].append(frames_similarity)

            # Get the number of lives
            if 'lives' in info:
                player['lives'].append(info['lives'])

            # Predict action
            action, _ = model.predict(observation, deterministic=True)
            player['action'].append(action)

            # step (transition) through the environment with the action
            # receiving the next observation, reward and if the episode has terminated or truncated
            observation, reward, terminated, truncated, info = env.step(action)
            player['reward'].append(reward)

            # If the episode has ended then we can reset to start a new episode
            if terminated or truncated:
                # win
                if reward == 1.0:
                    status = 2
                # lose
                elif (reward == -1.0) or (('lives' in info) and (info['lives'] == 0)):
                    status = 3
                player['status'].append(status)

                # End of the episode
                break
            else:
                if len(player["status"]) == 0:
                    status = 1
                else:
                    status = 0
                player['status'].append(status)

        # Check that all lists for the selected player have the same length
        lengths = {k: len(v) for k, v in player.items() if isinstance(v, list)}
        assert len(set(lengths.values())) == 1, (
            f"GameID: {shard}: List lengths are not equal: {lengths}"
        )

        # Yield examples for the selected player
        for step in range(len(player['frames'])):
            example = {
                "messages": {
                    "game": env_name,
                    "name": player_name,
                    "observation": None,  # Placeholder for observation information
                    "action_mask": None,  # Placeholder for action mask
                    "action": str(player['action'][step]),
                    "reward": player['reward'][step],
                    "status": player['status'][step],
                    "frames_similarity": player['frames_similarity'][step],
                    "info": None,  # Placeholder for additional information
                },
                "images": player['frames'][step],
            }
            yield example

    env.close()


def main():
    # Load player names for each environment
    with open("./sb3_agents/player_names.json", "r") as f:
        player_names = json.load(f)

    # Define the dataset features
    shards = list(range(NUM_GAMES))

    for env_name in env_names:
        dataset = Dataset.from_generator(
            dataset_generator,
            features=Features(
                {
                    "messages": {
                        "game": Value("string"),
                        "name": Value("string"),
                        "observation": Value("string"),
                        "action_mask": Value("string"),
                        "action": Value("string"),
                        "reward": Value("float32"),
                        "status": Value("int32"),
                        "frames_similarity": Sequence(Sequence(Value("float32"))),
                        "info": Value("string"),
                    },
                    "images": Sequence(Image()),
                }
            ),
            num_proc=40,
            gen_kwargs={
                "shards": shards,
                "env_name": env_name,
                "player_name": player_names[env_name],
            },
        )

        # Set metadata for the dataset
        dataset.info.description = "Arcade Learning Environment dataset"
        dataset.info.dataset_name = "Atari"
        dataset.info.citation = "https://ale.farama.org/environments/"
        dataset.info.license = "MIT License"
        dataset.info.homepage = "https://github.com/markub3327/SB3_Agents"
        dataset.info.version = Version("1.0.0")

        # Shuffle the dataset once before saving
        dataset = dataset.shuffle()
        print(f"Dataset shuffled!")

        # Save the dataset
        dataset.save_to_disk(
            f"/mnt/project/perun2601343/SB3_Agents/dataset/{env_name}",
            num_proc=8,
        )

if __name__ == "__main__":
    main()