import torch
import torch.nn as nn

from transformers import (
    AutoModel,
    PreTrainedModel,
    AutoModelForCausalLM,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from .Llava_config import HybridLlavaConfig

class MlpConnector(nn.Module):
    def __init__(self, image_size=1152, hidden_size=4608, output_size=2048):
        super(MlpConnector, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.model(x)
    
class HybridVisionModel(PreTrainedModel):
    config_class = HybridLlavaConfig

    def __init__(self, config: HybridLlavaConfig):
        super().__init__(config)
        self.config = config
        self.vision_model = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224")
        self.text_model = AutoModelForCausalLM.from_config(self.config.text_config)
        self.processor = SiglipImageProcessor().from_pretrained("google/siglip2-base-patch16-224")
        self.mlp_connector = MlpConnector(
            image_size=self.config.vision_config.hidden_size,
            hidden_size=self.config.vision_config.hidden_size * 4,
            output_size=self.config.text_config.hidden_size
        )
        # the first stage of Llava only train the connector
        # the second stage of Llava train the whole model except the vision model
        # so we need to freeze the vision model parameters all the time
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    @property
    def device(self):
        return self.text_model.device
    
    def image2tensor(self, image):
        if image.mode in ['RGBA', 'LA']:
            image = image.convert("RGB")
        image = self.processor(
            image = image,
            return_tensors="pt",
            do_resize=True,
            size={
                "height": 224, #self.config.vision_config.image_size,
                "width": 224, #self.config.vision_config.image_size
            },
        )["pixel_values"].to(device=self.vision_model.device, dtype=self.vision_model.dtype)

    def get_image_embeddings(self, image_tensor):
        with torch.no_grad():
            outputs = self.vision_model(pixel_values=image_tensor)
        image_embeds = outputs.last_hidden_state[:, 1:, :].squeeze() # remove cls token
        # # if bs = 1, remove batch dim, [bs, 196, 768] -> [196, 768]
        return image_embeds
        
    def count_vision_proj(self, tokens, inputs_embeds, vision_proj=None, seqlen=4096):
        # find_indices 的作用是找到 tokens 中与 image_ids 匹配的子序列的起始和结束索引。
        def find_image_indices(tokens, image_ids):
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            return {
                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                            matches[batch_idx].nonzero(as_tuple=True)[0]]
                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
            } or None

        image_indices = find_image_indices(tokens, self.config.image_ids)
        
        if not image_indices:
            return inputs_embeds
        
        if len(vision_proj.shape) == 3:
            vision_proj = vision_proj.unsqueeze(0) # [1, num_images, 196, 768]

        modified_inputs_embeds = inputs_embeds.clone()
        img_idx = 0
        for batch_idx, positions in image_indices.items():
            for _, (start_idx, end_idx) in enumerate(positions):
                if img_idx < vision_proj.size(1): # 确保 vision_proj 中有足够的图像嵌入
                    modified_inputs_embeds[batch_idx, start_idx:end_idx + 1] = vision_proj[batch_idx][img_idx]
                    img_idx += 1
                else:
                    break
        return modified_inputs_embeds
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        images: List = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        logits_to_keep: Optional[int] = 0,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
        
        if past_key_values is None:
            vision_tensors = None
            
            if images is not None:
                pixel_values = torch.stack([self.image2tensor(image) for image in images], dim=0)
            if pixel_values is not None:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                batch_size, img_num, channel, height, width = pixel_values.shape
                stack_dim = 1 if batch_size > 1 else 0 # if batch size > 1, stack along dim 1, else stack along dim 0
                vision_tensors = torch.stack(
                    [self.get_image_embeddings(pixel_values[:, i, :, :, :]) # [bs, 196, 768] if bs > 1, else [196, 768]
                     for i in range(img_num)],
                    dim=stack_dim
                ) # [bs, num_images, 196, 768] if bs > 1, else [num_images, 196, 768]
            
            if vision_tensors is not None:
                vision_proj = self.mlp_connector(vision_tensors)
                inputs_embeds = self.count_vision_proj(
                                        input_ids, 
                                        inputs_embeds, 
                                        vision_proj,
                                        input_ids.shape[1]
                                    )
        
        outputs = self.text_model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logits_to_keep=logits_to_keep,
        )
                
        return outputs


