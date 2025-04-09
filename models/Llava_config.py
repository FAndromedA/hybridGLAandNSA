from transformers import PretrainedConfig, SiglipVisionConfig
from .hybrid_config import HybridConfig

from typing import List

class HybridLlavaConfig(PretrainedConfig):
    model_type = "hybridLlava"

    def __init__(
            self,
            image_special_token: str = '<|reserved_special_token_1|>' * 256,
            image_ids: List = [128003] * 256,
            **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.text_config = HybridConfig(**kwargs.pop("text_config", {}))
        self.vision_config = SiglipVisionConfig(**kwargs.pop("vision_config", {}))
        super().__init__(**kwargs)
