import numpy as np
import torch
from gymnasium import spaces
from torch.distributions import Bernoulli, Categorical
from vocab import ids_action_vocab


def cumsum_with_reset(rewards, dones):
    cumsum = np.zeros_like(rewards, dtype=np.float32)
    score = np.zeros_like(rewards[0], dtype=np.float32)

def random_splits(
        states,
        actions,
        actions_logits,
        rewards,
        scores,
        terminated,
        truncated,
        started,
        lives,
        imgs_embed,
        *, min_size, max_size):
    # Check the shape of arrays before splitting
    assert (
        len(states) == len(actions) == len(actions_logits) == len(rewards) ==
        len(terminated) == len(truncated) == len(started) == len(lives) == len(imgs_embed)
    ), (
        f"Length mismatch in trajectory data: "
        f"states={len(states)}, actions={len(actions)}, "
        f"actions_logits={len(actions_logits)}, rewards={len(rewards)}, "
        f"terminated={len(terminated)}, truncated={len(truncated)}, "
        f"started={len(started)}, lives={len(lives)}, imgs_embed={len(imgs_embed)}"
    )

    indices = []
    current_idx = 0
    n_steps = len(states)
    remainder = False
    while current_idx < n_steps:
        remaining = n_steps - current_idx
        if remaining > min_size:
            current_idx += np.random.randint(
                min_size,
                min(remaining, max_size) + 1
            )
            indices.append(current_idx)
        else:
            remainder = True
            break

    # If there is a remainder, merge it into the final window
    indices = indices[:-1] if remainder else indices

    return (
        np.split(states, indices),
        np.split(actions, indices),
        np.split(actions_logits, indices),
        np.split(rewards, indices),
        np.split(scores, indices),
        np.split(terminated, indices),
        np.split(truncated, indices),
        np.split(started, indices),
        np.split(lives, indices),
        np.split(imgs_embed, indices),
    )


def rollout(
        vec_env,
        model,
        episode_length,
        *,
        img_embed_model=None,
        random=False
):
    state_list = []
    action_list = []
    action_logits_list = []
    reward_list = []
    score_list = []
    terminated_list = []
    truncated_list = []
    started_list = []
    lives_list = []
    imgs_embed_list = []

    action_space = vec_env.action_space

    ### Start of episode
    obs, info = vec_env.reset()
    score = np.zeros(vec_env.num_envs, dtype=np.float32)
    lives = np.array(
        [info[i]["lives"] if "lives" in info[0] else 0 for i in range(vec_env.num_envs)]
    )
    started = np.ones(vec_env.num_envs, dtype=np.bool)

    # Perform rollout
    for t in range(episode_length):
        if random:
            action = [vec_env.action_space.sample()] * vec_env.num_envs
        else:
            # Get the policy distribution and extract logits
            obs_tensor = torch.as_tensor(obs).to(model.device)
            with torch.no_grad():
                # Get features from the policy network
                features = model.policy.extract_features(obs_tensor)
                latent_pi = model.policy.mlp_extractor.forward_actor(features)
                logits = model.policy.action_net(latent_pi)
                action_logits_list.append(logits.cpu().numpy())

                if isinstance(action_space, spaces.Discrete):
                    action = torch.argmax(logits, dim=-1).cpu().numpy()     # hard labels
                elif isinstance(action_space, spaces.MultiBinary):
                    action = Bernoulli(logits=logits).sample().cpu().numpy()
                else:
                    raise ValueError()

        # Get state[t]
        rendered_img = vec_env.env_method("render")
        state_list.append(rendered_img)
        if img_embed_model:
            imgs_embed_list.append(img_embed_model.get_embedding(rendered_img))

        # Get action[t]
        action_list.append(np.asarray([
            ids_action_vocab[vec_env.env_name].inverse[action[i]] for i in range(vec_env.num_envs)
        ]))

        # Get lives[t]
        lives_list.append(lives)

        # Get score[t]
        score_list.append(score)

        # Get started[t]
        started_list.append(started)

        # Perform a step
        obs, reward, terminated, info = vec_env.step(action)
        print("Game: ", vec_env.env_name, ", t=", t, ": action:", action, "reward:", reward, "score:", score, "terminated:", terminated, "started", started, "info:", info)

        # Get reward[t] (reward for action taken)
        reward_list.append(reward.copy())

        # Get terminated[t] (terminated for action taken)
        terminated_list.append(terminated)

        # Get truncated[t] (truncated for action taken)
        truncated_list.append(
            [info[i]["TimeLimit.truncated"] for i in range(vec_env.num_envs)]
        )

        # Update lives[t+1]
        lives = np.array(
            [info[i]["lives"] if "lives" in info[0] else 0 for i in range(vec_env.num_envs)]
        )

        # Update score[t+1]
        end_of_game = np.logical_and(terminated, (lives < 1))
        score = np.where(end_of_game, 0.0, (score + reward))

        # Update started[t+1]
        started = end_of_game

    # Stack the Numpy arrays
    return (
        np.stack(state_list, axis=0),
        np.stack(action_list, axis=0),
        np.stack(action_logits_list, axis=0),
        np.stack(reward_list, axis=0),
        np.stack(score_list, axis=0),
        np.stack(terminated_list, axis=0),
        np.stack(truncated_list, axis=0),
        np.stack(started_list, axis=0),
        np.stack(lives_list, axis=0),
        np.stack(imgs_embed_list, axis=0) if img_embed_model else [],
    )