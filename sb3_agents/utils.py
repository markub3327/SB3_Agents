import numpy as np
import torch
import yaml
from gymnasium import spaces
from torch.distributions import Bernoulli
from transformers import AutoImageProcessor, AutoModel
from vocab import ids_action_vocab


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
    steps_list = []

    step_per_venv = np.asarray([0] * vec_env.num_envs)
    action_space = vec_env.action_space

    # Start of episode
    obs, info = vec_env.reset()
    score = np.zeros(vec_env.num_envs, dtype=np.float32)
    lives = np.array(
        [
            info[i]["lives"] if "lives" in info[0] else -1
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
        rendered_img = vec_env.env_method("render")
        state_list.append(rendered_img)
        if img_embed_model:
            imgs_embed_list.append(img_embed_model.get_embedding(rendered_img))

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
            ", step=",
            t,
            ": action:",
            action,
            "reward:",
            reward,
            "score:",
            score,
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

        # Get truncated[t] (truncated for action taken)
        truncated_list.append(
            [info[i]["TimeLimit.truncated"] for i in range(vec_env.num_envs)]
        )
        end_of_game = np.logical_and(terminated, (lives < 1))

        # Get time
        steps_list.append(step_per_venv)

        # Update time
        step_per_venv = np.where(end_of_game, 0, (step_per_venv + 1))

        # Update lives[t+1]
        lives = np.array(
            [
                info[i]["lives"] if "lives" in info[0] else -1
                for i in range(vec_env.num_envs)
            ]
        )

        # Update score[t+1]
        score = np.where(end_of_game, 0.0, (score + reward))

        # Update started[t+1]
        started = end_of_game

    # Stack the Numpy arrays
    return (
        state_list,
        action_list,
        action_logits_list,
        reward_list,
        score_list,
        terminated_list,
        truncated_list,
        started_list,
        lives_list,
        imgs_embed_list,
        steps_list,
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
