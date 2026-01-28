import numpy as np
import torch
from gymnasium import spaces
from torch.distributions import Bernoulli, Categorical
from vocab import ids_action_vocab


def cumsum_with_reset(rewards, dones):
    cumsum = np.zeros_like(rewards, dtype=np.float32)
    score = np.zeros_like(rewards[0], dtype=np.float32)

    for i in range(rewards.shape[0]):
        score += rewards[i]
        cumsum[i] = score
        score = np.where(dones[i], 0.0, score)

    return cumsum


def rollout(vec_env, model, *, random=False):
    states = []
    actions = []
    actions_logits = []
    rewards = []
    terminated = []
    truncated = []
    lives = []

    action_space = vec_env.action_space
    obs = vec_env.reset()
    n_envs = obs.shape[0]
    temperature = 0.7
    episode_length = 8192
    for t in range(episode_length):
        if random:
            action = vec_env.action_space.sample()
        else:
            # Get the policy distribution and extract logits
            obs_tensor = torch.as_tensor(obs).to(model.device)
            with torch.no_grad():
                # Get features from the policy network
                features = model.policy.extract_features(obs_tensor)
                latent_pi = model.policy.mlp_extractor.forward_actor(features)
                logits = model.policy.action_net(latent_pi)

                if isinstance(action_space, spaces.Discrete):
                    action = (
                        Categorical(logits=logits / temperature).sample().cpu().numpy()
                    )  # soft labels
                elif isinstance(action_space, spaces.MultiBinary):
                    action = Bernoulli(logits=logits).sample().cpu().numpy()
                else:
                    raise ValueError()

        rendered_img = vec_env.env_method("render")
        obs, reward, done, info = vec_env.step(action)
        action = [
            ids_action_vocab[vec_env.env_name].inverse[action[i]] for i in range(n_envs)
        ]
        # print("action:", action, "reward:", reward, "terminated:", done, "info:", info)

        states.append(rendered_img)
        actions.append(action)
        actions_logits.append(logits.cpu().numpy())
        rewards.append(reward.copy())
        terminated.append(done.copy())
        truncated.append([info[i]["TimeLimit.truncated"] for i in range(n_envs)])
        lives.append(
            [info[i]["lives"] if "lives" in info[0] else 0 for i in range(n_envs)]
        )

    # Stack the Numpy arrays
    states = np.stack(states, axis=0)
    actions = np.stack(actions, axis=0)
    actions_logits = np.stack(actions_logits, axis=0)
    rewards = np.stack(rewards, axis=0)
    terminated = np.stack(terminated, axis=0)
    truncated = np.stack(truncated, axis=0)
    lives = np.stack(lives, axis=0)
    end_of_game = np.logical_and(terminated, (lives <= 0))

    return (
        states,
        actions,
        actions_logits,
        cumsum_with_reset(rewards, end_of_game),
        terminated,
        truncated,
        lives,
    )
