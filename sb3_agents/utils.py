import torch
import yaml
from transformers import AutoImageProcessor, AutoModel


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

def load_hyperparams(env_name, file_path):
    """
    Loads hyperparameters for a specific environment from a YAML file.
    """
    with open(file_path, "r") as f:
        all_params = yaml.load(f, Loader=yaml.SafeLoader)

    if env_name not in all_params:
        raise ValueError(f"Environment '{env_name}' not found in hyperparameters list")

    return all_params[env_name]