import numpy as np
import torch
import yaml
from gymnasium import spaces
from gymnasium.spaces import Box
from torch.distributions import Bernoulli
from transformers import AutoImageProcessor, AutoModel
from vocab import ids_action_vocab
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations


class ImageFilterForQueue:
    _model_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"  # ViT-0.3B (distilled)

    def __init__(self):
        super().__init__()
        # Image filter for redundant images in queue (DinoV3 Vision Transformer model)
        self.processor = AutoImageProcessor.from_pretrained(self._model_id)
        self.model = AutoModel.from_pretrained(self._model_id, device_map="cuda")

    def get_embedding(self, inputs):
        with torch.no_grad():  # Don't store gradients
            inputs = self.processor(images=inputs, return_tensors="pt").to(
                self.model.device
            )
            outputs = self.model(**inputs)
        return outputs.pooler_output.cpu().numpy()


def rollout(vec_env, model, *, episode_length, n_stack, img_embed_model=None, random=False):
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

    # Start of episode
    obs, info = vec_env.reset()
    rendered_img = vec_env.env_method("render")
    stacked_obs =  StackedObservations(
        vec_env.num_envs,
        n_stack,
        Box(0, 255, rendered_img[0].shape, dtype=np.uint8),
    )
    rendered_img = stacked_obs.reset(
        np.asarray(rendered_img, dtype=np.uint8)
    )
    rendered_img = np.stack(np.split(rendered_img, n_stack, axis=-1), axis=1)
    score = np.zeros(vec_env.num_envs, dtype=np.float32)
    lives = np.array(
        [
            info[i]["lives"] if "lives" in info[0] else None
            for i in range(vec_env.num_envs)
        ]
    )
    started = np.ones(vec_env.num_envs, dtype=np.bool)

    # Perform rollout
    for _ in range(episode_length):
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
                    action = torch.argmax(logits, dim=-1).cpu().numpy()  # hard labels
                elif isinstance(action_space, spaces.MultiBinary):
                    action = Bernoulli(logits=logits).sample().cpu().numpy()
                else:
                    raise ValueError()

        # Get state[t]
        state_list.append(rendered_img)
        if img_embed_model:
            img_batch = rendered_img.reshape(-1, rendered_img.shape[2], rendered_img.shape[3], rendered_img.shape[4])
            print(img_batch.shape)
            img_embed = img_embed_model.get_embedding(img_batch)
            img_embed = np.stack(np.split(img_embed, vec_env.num_envs, axis=0), axis=0)
            print(img_embed.shape)
            imgs_embed_list.append(img_embed)

        # Get action[t]
        action_list.append(
            np.asarray(
                [
                    ids_action_vocab[vec_env.env_name].inverse[action[i]]
                    for i in range(vec_env.num_envs)
                ]
            )
        )

        # Get lives[t]
        lives_list.append(lives)

        # Get score[t]
        score_list.append(score)

        # Get started[t]
        started_list.append(started)

        # Perform a step
        obs, reward, terminated, info = vec_env.step(action)
        print(
            "Game: ",
            vec_env.env_name,
            "action:",
            action,
            "reward:",
            reward,
            "score:",
            score,
            "lives:",
            lives,
            "terminated:",
            terminated,
            "started",
            started,
            "info:",
            info,
        )

        # Get reward[t] (reward for action taken)
        reward_list.append(reward.copy())

        # Get terminated[t] (terminated for action taken)
        terminated_list.append(terminated)
        for l in lives:
            if l:
                end_of_game = np.logical_and(terminated, (lives < 1))
            else:
                end_of_game = terminated

        # Get truncated[t] (truncated for action taken)
        truncated_list.append(
            [info[i]["TimeLimit.truncated"] for i in range(vec_env.num_envs)]
        )

        # Update lives[t+1]
        lives = np.array(
            [
                info[i]["lives"] if "lives" in info[0] else None
                for i in range(vec_env.num_envs)
            ]
        )

        # Update state[t+1]
        rendered_img = vec_env.env_method("render")
        rendered_img, _ = stacked_obs.update(
            np.asarray(rendered_img, dtype=np.uint8),
            terminated,
            ([{}] * vec_env.num_envs)
        )
        rendered_img = np.stack(np.split(rendered_img, n_stack, axis=-1), axis=1)

        # Update score[t+1]
        score = np.where(end_of_game, 0.0, (score + reward))

        # Update started[t+1]
        started = end_of_game

    # Stack the Numpy arrays
    return (
        np.stack(state_list, axis=0),
        np.stack(action_list, axis=0),
        np.stack(action_logits_list, axis=0) if not random else [],
        np.stack(reward_list, axis=0),
        np.stack(score_list, axis=0),
        np.stack(terminated_list, axis=0),
        np.stack(truncated_list, axis=0),
        np.stack(started_list, axis=0),
        np.stack(lives_list, axis=0),
        np.stack(imgs_embed_list, axis=0) if img_embed_model else [],
    )


def load_hyperparams(env_name, file_path="./sb3_agents/hyperparams.yml"):
    """
    Loads hyperparameters for a specific environment from a YAML file.
    """
    with open(file_path, "r") as f:
        # Loader=yaml.SafeLoader is recommended for security
        all_params = yaml.load(f, Loader=yaml.SafeLoader)

    if env_name not in all_params:
        raise ValueError(f"Environment '{env_name}' not found in hyperparameters list")

    return all_params[env_name]
