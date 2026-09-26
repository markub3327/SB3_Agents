import datetime
import minigrid
from minigrid.wrappers import FlatObsWrapper
import gymnasium
import os
import json
import hashlib
import numpy as np
from stable_baselines3 import PPO
from datasets import Dataset, Features, Value, Image, Version, Sequence
from collections import deque

# Use a dummy audio driver
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Metadata
NUM_GAMES = 1_000_000

### Minigrid
env_names = (
    "MiniGrid-UnlockPickup-v0",
    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-GoToObject-6x6-N2-v0",
    "MiniGrid-Dynamic-Obstacles-Random-5x5-v0",
    "MiniGrid-SimpleCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-Unlock-v0",
    "MiniGrid-KeyCorridorS3R1-v0",
    "MiniGrid-RedBlueDoors-6x6-v0",
    "MiniGrid-GoToDoor-5x5-v0",
    "MiniGrid-Fetch-5x5-N2-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-Empty-Random-5x5-v0",
)

def get_ascii_map(base_env):
    symbols = {
        "wall": "#",
        "floor": ".",
        "key": "K",
        "ball": "B",
        "box": "X",
        "goal": "G",
        "lava": "L",
    }

    agent_symbols = [
        ">",  # 0: Right
        "v",  # 1: Down
        "<",  # 2: Left
        "^",  # 3: Up
    ]

    cell_width = 8
    border = "+" + (("-" * (cell_width + 2)) + "+") * base_env.width
    obs = border + "\n"

    for y in range(base_env.height):
        obs += "|"

        for x in range(base_env.width):
            if tuple(base_env.agent_pos) == (x, y):
                cell_content = agent_symbols[base_env.agent_dir]
            else:
                cell = base_env.grid.get(x, y)

                if cell is None:
                    cell_content = ""
                else:
                    if cell.type == "door":
                        if cell.is_open:
                            cell_symbol = "/"  # State 0: open
                        elif cell.is_locked:
                            cell_symbol = "D"  # State 2: locked
                        else:
                            cell_symbol = "d"  # State 1: closed and unlocked
                    else:
                        cell_symbol = symbols[cell.type]

                    cell_content = f"{cell_symbol}:{cell.color}"

            obs += f" {cell_content:<{cell_width}} |"

        obs += "\n" + border + "\n"

    return obs

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
            env = gymnasium.make(env_name, render_mode="rgb_array")
            env = FlatObsWrapper(env)
            for shard in shards:
                # Initialize the frame queue
                frame_queue = deque(maxlen=2)

                # Reset the environment for a new game
                observation, _ = env.reset()

                # Init agent
                player = {
                    "frames": [],
                    "observation": [],
                    "action": [],
                    "reward": [],
                    "status": [],
                    "info": f"Mission: {env.unwrapped.mission}",
                }

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
                    obs = get_ascii_map(env.unwrapped)
                    player["observation"].append(obs)

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
                            if reward >= 0.9:
                                status = 2
                            # lose
                            else:
                                status = 3
                        player['status'].append(status)                       
                        break
                    else:
                        if len(player["status"]) == 0:
                            player["status"].append(1)
                        else:
                            player["status"].append(0)
 
                print(f"Length of player['frames']: {len(player['frames'])}")
                print(f"Length of player['observation']: {len(player['observation'])}")
                print(f"Length of player['action']: {player['action']}")
                print(f"Length of player['reward']: {player['reward']}")
                print(f"Length of player['status']: {player['status']}")

                # Check that all lists for the selected player have the same length
                lengths = {k: len(v) for k, v in player.items() if isinstance(v, list)}
                assert len({length for name, length in lengths.items() if not name == "info"}) == 1, (
                    f"GameID: {shard}: List lengths are not equal: {lengths}"
                )

                if player["status"][-1] != 3 and player["status"][-1] != 5:
                    # Yield examples for the selected player
                    for step in range(len(player['observation'])):
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
                                "info": player['info'],
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
            # num_proc=32,
            gen_kwargs={"shards": shards},
        )

        # Set metadata for the dataset
        dataset.info.description = "MiniGrid dataset"
        dataset.info.dataset_name = "MiniGrid"
        dataset.info.citation = "https://minigrid.farama.org/environments/minigrid/"
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
            # num_proc=32,
        )

if __name__ == "__main__":
    main()