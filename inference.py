#!/usr/bin/env python
# coding: utf-8

# In[10]:


import json
import random
from pathlib import Path
import torch
import ale_py
import cv2
import numpy as np
from bidict import bidict
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from torch.distributions import Categorical


# In[ ]:

ids_action_vocab = bidict(
    {
        "noop": 0,
        "fire": 1,
        "up": 2,
        "right": 3,
        "left": 4,
        "down": 5,
        "upright": 6,
        "upleft": 7,
        "downright": 8,
        "downleft": 9,
        "upfire": 10,
        "rightfire": 11,
        "leftfire": 12,
        "downfire": 13,
        "uprightfire": 14,
        "upleftfire": 15,
        "downrightfire": 16,
        "downleftfire": 17,
        "reset": 40,
    }
)

# In[11]:

env_names = [
    "BoxingNoFrameskip-v4",  # ok, OK
    "DemonAttackNoFrameskip-v4",  # ok
    "FishingDerbyNoFrameskip-v4",  # ok
    "FreewayNoFrameskip-v4",  # ok
    "GopherNoFrameskip-v4",  # ok
    "KrullNoFrameskip-v4",  # ok
    "KungFuMasterNoFrameskip-v4",  # ok
    "PongNoFrameskip-v4",  # ok
    "BreakoutNoFrameskip-v4", # ok
    "AssaultNoFrameskip-v4",  # ok, OK
    "VideoPinballNoFrameskip-v4",  # ok
    "StarGunnerNoFrameskip-v4",  # ok
    "CrazyClimberNoFrameskip-v4",  # ok, OK
    "AtlantisNoFrameskip-v4",  # ok
    "EnduroNoFrameskip-v4",   # ok
    "NameThisGameNoFrameskip-v4",  # ok
    "RoadRunnerNoFrameskip-v4",  # ok
    "TutankhamNoFrameskip-v4",   # ok
    "BankHeistNoFrameskip-v4",  # ok, OK
    "DoubleDunkNoFrameskip-v4", # ok
    "JamesbondNoFrameskip-v4",  # ok
    "KangarooNoFrameskip-v4",  # ok
    "QbertNoFrameskip-v4",  # ok
    "UpNDownNoFrameskip-v4",
]

# env_names = [
#     "BoxingNoFrameskip-v4",  # ok, OK
#     # "DemonAttackNoFrameskip-v4",  # ok
#     "FishingDerbyNoFrameskip-v4",  # ok
#     "FreewayNoFrameskip-v4",  # ok
#     "GopherNoFrameskip-v4",  # ok
#     "KrullNoFrameskip-v4",  # ok
#     "KungFuMasterNoFrameskip-v4",  # ok
#     "PongNoFrameskip-v4",  # ok
#     # "BreakoutNoFrameskip-v4", # ok
#     "AssaultNoFrameskip-v4",  # ok, OK
#     # "VideoPinballNoFrameskip-v4",  # ok
#     "StarGunnerNoFrameskip-v4",  # ok
#     "CrazyClimberNoFrameskip-v4",  # ok, OK
#     # "AtlantisNoFrameskip-v4",  # ok
#     # "EnduroNoFrameskip-v4",   # ok
#     # "NameThisGameNoFrameskip-v4",  # ok
#     "RoadRunnerNoFrameskip-v4",  # ok
#     # "TutankhamNoFrameskip-v4",   # ok
#     "BankHeistNoFrameskip-v4",  # ok, OK
#     # "DoubleDunkNoFrameskip-v4", # ok
#     "JamesbondNoFrameskip-v4",  # ok
#     "KangarooNoFrameskip-v4",  # ok
#     "QbertNoFrameskip-v4",  # ok
#     # "UpNDownNoFrameskip-v4",
# ]


def save_trajectory(*, part_id, states, actions, action_logits, rewards, terminated, truncated, lives, env_name, init_reward):
    print(f"Saving the part {part_id}.")
    ds_path = f"../dataset/{env_name}_part{part_id}.npz"
    Path(ds_path).parent.mkdir(parents=True, exist_ok=True)

    if init_reward is not None:
        rewards[0] += init_reward

    states = np.asarray(states)
    actions = np.asarray(actions)
    action_logits = np.asarray(action_logits)
    rewards = np.cumsum(rewards, axis=-1)
    terminated = np.asarray(terminated)
    truncated = np.asarray(truncated)
    lives = np.asarray(lives)

    # Store on disk
    np.savez_compressed(
        ds_path,
        name=env_name,
        states=states,
        actions=actions,
        action_logits=action_logits,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        lives=lives,
    )

    return rewards[-1]


# In[12]:

for env_name in env_names:
    print(f"Generating the dataset for {env_name} environment.")

    # There already exists an environment generator
    # that will make and wrap atari environments correctly.
    # Here we are also multi-worker training (n_envs=4 => 4 environments)
    vec_env = make_atari_env(env_name, n_envs=1, seed=1234, wrapper_kwargs={"clip_reward": False})
    # Frame-stacking with 4 frames
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model = PPO.load(f"save/{env_name}/best_model.zip", env=vec_env, custom_objects={"learning_rate": lambda _: 0.0})

    # Recorder
    #
    # Video from the agent trajectory must be divided into segments, where the agent does only perfect action (responses to the environment).
    #
    states = []
    actions = []
    action_logits = []
    rewards = []
    terminated = []
    truncated = []
    lives = []

    obs = vec_env.reset()
    part_id, sample_counter, last_reward = 1, 0, 0
    temperature = 0.6
    for t in range(15_000):
        # Get the policy distribution and extract logits
        obs_tensor = torch.as_tensor(obs).to(model.device).permute(0, 3, 1, 2)
        with torch.no_grad():
            # Get features from the policy network
            features = model.policy.extract_features(obs_tensor)
            latent_pi = model.policy.mlp_extractor.forward_actor(features)
            logits = model.policy.action_net(latent_pi)
            action = Categorical(logits=logits / temperature).sample().cpu().numpy()  # soft labels

        rendered_img = vec_env.render("rgb_array")
        obs, reward, done, info = vec_env.step(action)
        action = ids_action_vocab.inverse[action[0]]
        has_lives = info[0]["lives"] > 0
        sample_counter += 1
        # print("action:", action, "reward:", reward, "terminated:", done, "info:", info)

        states.append(rendered_img.copy())
        actions.append(action)
        action_logits.append(logits[0].cpu().numpy())
        rewards.append(reward[0].copy())
        terminated.append(done[0].copy())
        truncated.append(info[0]["TimeLimit.truncated"])
        lives.append(info[0]["lives"])

        # Create a part of dataset
        if t > 0 and (t % 1000) == 0:
            last_reward = save_trajectory(
                part_id=part_id,
                states=states,
                actions=actions,
                action_logits=action_logits,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                lives=lives,
                env_name=env_name,
                init_reward=last_reward,
            )
            part_id = part_id + 1
            # Free the RAM memory
            states.clear()
            actions.clear()
            action_logits.clear()
            rewards.clear()
            terminated.clear()
            truncated.clear()
            lives.clear()

        # End the episode before saving
        if done and not has_lives:
            break

    # Save to one file with multiple arrays
    score = save_trajectory(
        part_id=part_id,
        states=states,
        actions=actions,
        action_logits=action_logits,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        lives=lives,
        env_name=env_name,
        init_reward=last_reward,
    )

    # Wrong reward cumulation
    if "episode" in info[0]:
        assert score == info[0]["episode"]["r"], f"For {env_name} the cumulative rewards {score} is not equal to {info[0]["episode"]["r"]} !!!"
    else:
        print("Reward cannot be compared with info dict !!!")

    print("Episode finished with reward:", score)
    print("Episode finished with lives:", lives[-1])
    print(f"Generated was {sample_counter} samples.")
