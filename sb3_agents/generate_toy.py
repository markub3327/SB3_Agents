import gymnasium
import os
import json
import hashlib
import numpy as np
from stable_baselines3 import PPO
from datasets import Dataset, Features, Value, Image, Version, List
from collections import deque
from gymnasium.wrappers import FlattenObservation
from gymnasium.envs.toy_text.blackjack import sum_hand, usable_ace, is_bust


# Use a dummy audio driver
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Metadata
NUM_GAMES = 100_000  # Total number of games to generate

### Toy Text
env_names = (
    "Blackjack-v1",
    "FrozenLake-v1",
)


def dataset_generator(shards, env_name, player_name):
    print(f"Generating the dataset for {env_name} environment.")

    # Load pre-trained model
    model = PPO.load(
        f"./save/{env_name}/best_model.zip",
        custom_objects={"learning_rate": lambda _: 0.0, "_last_obs": None},
    )
    
    # Init environment
    if "FrozenLake" in env_name:
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
        }

        # Reset the environment for a new game
        observation, _ = env.reset()

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

            # Encode the current state
            if "Blackjack" in env_name:
                obs = f"My hand has a value of {sum_hand(env.unwrapped.player)}. The dealer's visible card has a value of {env.unwrapped.dealer[0]}."
                if usable_ace(env.unwrapped.player):
                    obs += " My hand contains a usable Ace, which is counted as 11."
                else:
                    obs += " My hand does not contain a usable Ace."
                obs += "\n"
            elif "FrozenLake" in env_name:
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
            observation, reward, terminated, truncated, _ = env.step(action)
            player['reward'].append(reward)

            # If the episode has ended then we can reset to start a new episode
            if terminated or truncated:
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
                    "observation": player['observation'][step],
                    "action_mask": None,  # Placeholder for action mask
                    "action": str(player['action'][step]),
                    "reward": player['reward'][step],
                    "status": player['status'][step],
                    "frames_similarity": None,  # Placeholder for frames similarity information
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
                        "frames_similarity": List(Value("float32")),
                        "info": Value("string"),
                    },
                    "images": List(Image()),
                }
            ),
            num_proc=48,
            gen_kwargs={
                "shards": shards,
                "env_name": env_name,
                "player_name": player_names[env_name],
            },
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
        print("Total samples:", len(dataset))

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