import torch
import yaml
import gymnasium
import stable_retro as retro
from transformers import AutoImageProcessor, AutoModel
from minigrid.wrappers import FlatObsWrapper
from gymnasium.wrappers import TimeLimit, FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    MaxAndSkipEnv,
    WarpFrame,
)


class FrameFilterForQueue:
    # DinoV3 Vision Transformer model
    _model_id = "facebook/dinov3-vith16plus-pretrain-lvd1689m"  # ViT-0.8B

    def __init__(self):
        super().__init__()
        # Image filter for redundant images in queue
        self.processor = AutoImageProcessor.from_pretrained(self._model_id)
        self.model = AutoModel.from_pretrained(self._model_id, device_map="auto")

    def _get_embedding(self, images):
        if not isinstance(images, list):
            images = list(images)
        inputs = self.processor(images=images, return_tensors="pt").to(
            self.model.device
        )
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs.pooler_output

    def get_similarity(self, images):
        imgs_embed = self._get_embedding(images)
        sims = torch.cosine_similarity(imgs_embed, imgs_embed)
        return sims.cpu().numpy()

def load_hyperparams(env_name, file_path):
    """
    Loads hyperparameters for a specific environment from a YAML file.
    """
    with open(file_path, "r") as f:
        all_params = yaml.load(f, Loader=yaml.SafeLoader)

    if env_name not in all_params:
        raise ValueError(f"Environment '{env_name}' not found in hyperparameters list")

    return all_params[env_name]

def make_retro_env(env_name):
    def _body():
        """
        Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
        """
        env = retro.make(env_name, retro.State.DEFAULT, render_mode="rgb_array")
        env = TimeLimit(env, max_episode_steps=8192)
        env = Monitor(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env, width=96, height=96)
        env = ClipRewardEnv(env)
        return env

    return _body

def make_minigrid_env(env_name):
    def _body():
        env = gymnasium.make(env_name, render_mode="rgb_array")
        env = FlatObsWrapper(env)
        return env

    return _body

def make_toy_text_env(env_name):
    def _body():
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
        return env

    return _body
