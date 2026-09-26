import torch
import yaml
from transformers import AutoImageProcessor, AutoModel


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