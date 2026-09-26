import datetime
import gymnasium
import os
import json
import hashlib
import numpy as np
from stable_baselines3 import PPO
from utils import FrameFilterForQueue
from datasets import Dataset, Features, Value, Image, Version, Sequence
from collections import deque
from gymnasium.wrappers import FlattenObservation
from gymnasium.envs.toy_text.blackjack import sum_hand, usable_ace, is_bust

# Use a dummy audio driver
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Metadata
NUM_GAMES = 1_000_000

### Toy Text
env_names = (
    "Blackjack-v1",
    "FrozenLake-v1",
)


def main():
    # Load player names for each environment
    with open("./sb3_agents/player_names.json", "r") as f:
        player_names = json.load(f)

    for env_name in env_names:
        def dataset_generator(shards):
            print(f"Generating the dataset for {env_name} environment.")

            # Load pre-trained model
            model = PPO.load(
                f"./save/{env_name}/best_model.zip",
                custom_objects={"learning_rate": lambda _: 0.0, "_last_obs": None},
            )
            
            # Init environment
            if env_name == "FrozenLake-v1":
                env = gymnasium.make(
                    env_name,
                    render_mode="rgb_array",
                    reward_schedule=(1, -1, 0),
                    is_slippery=False,
                    map_name="8x8",
                )
            else:
                env = gymnasium.make(env_name, render_mode="rgb_array")
            env = FlattenObservation(env)
            for shard in shards:
                # Initialize the frame queue
                frame_queue = deque(maxlen=2)
        
                # Init agent
                player = {
                    "frames": [],
                    "observation": [],
                    "action": [],
                    "reward": [],
                    "status": [],
                    "info": [],
                }

                # Reset the environment for a new game
                observation, _ = env.reset()

                # The initial frame
                frame = env.render()
                frame_queue.append(frame)

                # Run the game loop
                yield_episode = True
                while True:
                    # Render the current frame
                    frame = env.render()
                    frame_queue.append(frame)
                    player['frames'].append(
                        np.stack(frame_queue, axis=0)
                    )

                    # Encode the current state
                    if env_name == "Blackjack-v1":
                        obs = f"The player's hand has a total value of {sum_hand(env.unwrapped.player)}. The dealer's visible card has a value of {env.unwrapped.dealer[0]}."
                        if usable_ace(env.unwrapped.player) == 0:
                            obs += " The hand does not contain a usable Ace."
                        else:
                            obs += " The hand contains a usable Ace that is counted as 11."
                        obs += "\n"
                    elif env_name == "FrozenLake-v1":
                        desc = [[c.decode("utf-8") for c in line] for line in env.unwrapped.desc]
                        row, col = env.unwrapped.s // env.unwrapped.ncol, env.unwrapped.s % env.unwrapped.ncol
                        desc[row][col] = "P"
                        obs = "+" + ("-" * 3 + "+") * env.unwrapped.ncol + "\n"
                        for map_row in desc:
                            obs += "| "
                            for cell in map_row:
                                obs += cell + " | "
                            obs += "\n"
                            obs += "+" + ("-" * 3 + "+") * env.unwrapped.ncol + "\n"
                    player['observation'].append(obs)

                    # Predict action
                    action, _ = model.predict(observation, deterministic=True)
                    action = int(action)
                    player['action'].append(action)

                    # step (transition) through the environment with the action
                    # receiving the next observation, reward and if the episode has terminated or truncated
                    observation, reward, terminated, truncated, info = env.step(action)
                    player['reward'].append(reward)

                    # If the episode has ended then we can reset to start a new episode
                    if terminated or truncated:
                        if truncated:
                            status = 5
                        else:
                            # win
                            if reward == 1.0:
                                status = 2
                            # lose
                            elif reward == -1.0:
                                status = 3
                            # draw
                            else:
                                status = 4
                        player['status'].append(status)
                        if env_name == "Blackjack-v1":
                            if action == 1 and is_bust(env.unwrapped.player):
                                player['info'].append(f"The player's total hand value of {sum_hand(env.unwrapped.player)} exceeds 21, resulting in a bust and immediate loss.")
                            else:
                                player['info'].append(f"The dealer's final hand has a total value of {sum_hand(env.unwrapped.dealer)}.")
                        break
                    else:
                        if len(player["status"]) == 0:
                            player["status"].append(1)
                        else:
                            player["status"].append(0)
                        if env_name == "Blackjack-v1":
                            player['info'].append(None)

                print(f"Length of player['frames']: {len(player['frames'])}")
                print(f"Length of player['observation']: {len(player['observation'])}")
                print(f"Length of player['action']: {player['action']}")
                print(f"Length of player['reward']: {player['reward']}")
                print(f"Length of player['status']: {player['status']}")
                print(f"Length of player['info']: {player['info']}")

                # Check that all lists for the selected player have the same length
                lengths = {k: len(v) for k, v in player.items() if isinstance(v, list)}
                assert len({length for name, length in lengths.items() if not (name == "info" and length == 0)}) == 1, (
                    f"GameID: {shard}: List lengths are not equal: {lengths}"
                )

                # if not find the goal
                if env_name == "FrozenLake-v1":
                    yield_episode = (player["reward"][-1] == 1.0)

                if yield_episode:
                    # Yield examples for the selected player
                    for step in range(
                        len(player['frames'])
                    ):
                        example = {
                            "messages": {
                                "game": env_name,
                                "name": player_names[env_name],
                                "observation": player['observation'][step],
                                "action_mask": None,  # Placeholder for action mask
                                "action": str(player['action'][step]),
                                "reward": player['reward'][step],
                                "status": player['status'][step],
                                "lives": None,  # Placeholder for lives information
                                "frames_similarity": None,  # Placeholder for frames similarity information
                                "info": player['info'][step] if env_name == "Blackjack-v1" else None,
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
                        "status": Value("int32"),
                        "lives": Value("int32"),
                        "frames_similarity": Sequence(Sequence(Value("float32"))),
                        "info": Value("string"),
                    },
                    "images": Sequence(Image()),
                }
            ),
            num_proc=32,
            gen_kwargs={"shards": shards},
        )

        # Set metadata for the dataset
        dataset.info.description = "Toy Text dataset"
        dataset.info.dataset_name = "Toy Text"
        dataset.info.citation = "https://ale.farama.org/environments/"
        dataset.info.license = "MIT License"
        dataset.info.homepage = "https://github.com/markub3327/SB3_Agents"
        dataset.info.version = Version("1.0.0")

        # Remove duplicate samples based on their hash
        keep_indices = {}
        for i, obs in enumerate(dataset["messages"]["observation"]):
            h = hashlib.sha3_512(obs.encode("utf-8")).hexdigest()
            if h not in keep_indices:
                keep_indices[h] = i
            else:
                print(f"Duplicate observation found for {i} sample!")
        dataset = dataset.select(list(keep_indices.values()))

        # Save the dataset
        dataset.save_to_disk(
            f"/mnt/project/perun2601343/SB3_Agents/dataset/{env_name}",
            num_proc=min(32, max(1, len(dataset))),
        )

if __name__ == "__main__":
    main()