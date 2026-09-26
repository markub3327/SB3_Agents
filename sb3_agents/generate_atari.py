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

)


def main():
    # Initialize the frame filter
    img_filter = FrameFilterForQueue()

    # Load player names for each environment
    with open("./sb3_agents/player_names.json", "r") as f:
        player_names = json.load(f)

    for env_name in env_names:
        print(f"Generating the dataset for {env_name} environment.")

        # Load pre-trained model
        model = PPO.load(
            f"./save/{env_name}/best_model.zip",
            custom_objects={"learning_rate": lambda _: 0.0, "_last_obs": None},
        )

        def dataset_generator(shards):
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
                    "truncated": [],
                    "lives": [],
                    "status": [1],
                    "frames_similarity": [],
                }

                # Reset the environment for a new game
                observation, info = env.reset()
                print("info", info)

                # The initial frame
                frame = env.render()
                frame_queue.append(np.zeros_like(frame))  # Add a zero frame for the initial observation

                # Run the game loop
                # while True:
                for step in range(1000):
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
                    player['truncated'].append(truncated)

                    # If the episode has ended then we can reset to start a new episode
                    if terminated or truncated:
                        # Win
                        if reward == 1.0:
                            players[agent]['status'].append(2)
                        # Lose
                        elif reward == -1.0:
                            players[agent]['status'].append(3)
                        # Draw
                        else:
                            players[agent]['status'].append(4)
                        break
                    else:
                        player['status'].append(0)

            print(f"Length of player['frames']: {len(player['frames'])}")
            print(f"Length of player['action']: {len(player['action'])}")
            print(f"Length of player['reward']: {len(player['reward'])}")
            print(f"Length of player['truncated']: {len(player['truncated'])}")
            print(f"Length of player['status']: {len(player['status'])}")
            print(f"Length of player['lives']: {len(player['lives'])}")
            print(f"Length of player['frames_similarity']: {len(player['frames_similarity'])}")

            # Yield examples for the selected player
            for step in range(
                len(player['frames'])
            ):
                example = {
                    "messages": {
                        "game": env_name,
                        "name": player_names.get(env_name, None),
                        "observation": None,  # Placeholder for observation information
                        "action_mask": None,  # Placeholder for action mask
                        "action": str(player['action'][step]),
                        "reward": player['reward'][step],
                        "truncated": player['truncated'][step],
                        "status": player['status'][step],
                        "lives": player['lives'][step] if len(player['lives']) > 0 else None,
                        "frames_similarity": player['frames_similarity'][step],
                    },
                    "images": player['frames'][step],
                }

                yield example

            env.close()

        # Define the dataset features
        shards = list(range(NUM_GAMES))
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
                        "truncated": Value("bool"),
                        "status": Value("int32"),
                        "lives": Value("int32"),
                        "frames_similarity": Sequence(Sequence(Value("float32"))),
                    },
                    "images": Sequence(Image()),
                }
            ),
            # num_proc=32,
            gen_kwargs={"shards": shards},
        )

        # Set metadata for the dataset
        dataset.info.description = "Arcade Learning Environment dataset"
        dataset.info.dataset_name = "Atari"
        dataset.info.citation = "https://ale.farama.org/environments/"
        dataset.info.license = "MIT License"
        dataset.info.homepage = "https://github.com/markub3327/SB3_Agents"
        dataset.info.version = Version("1.0.0")

        # Save the dataset
        dataset.save_to_disk(
            f"/mnt/project/perun2601343/SB3_Agents/dataset/{env_name}",
            # num_proc=32,
        )

if __name__ == "__main__":
    main()