
import math
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.nsa import NativeSparseAttention
from fla.layers.attn import Attention

from fla.layers.gla import GatedLinearAttention
from fla.models.gla.configuration_gla import GLAConfig
from .hybrid_config import HybridConfig
from fla.models.utils import Cache
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules import GatedMLP
from fla.modules import RMSNorm

# if TYPE_CHECKING:
from transformers.processing_utils import Unpack

class HybridBlock(nn.Module):
    def __init__(self,  config: HybridConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = NativeSparseAttention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                head_dim=config.attn['head_dim'],
                qkv_bias=config.attn['qkv_bias'],
                block_size=config.attn['block_size'],
                block_counts=config.attn['block_counts'],
                window_size=config.attn['window_size'],
                rope_theta=config.attn['rope_theta'],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx
            )
        else:
            self.attn = GatedLinearAttention(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                feature_map=config.feature_map,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                use_output_gate=config.use_output_gate,
                gate_fn=config.hidden_act,
                elementwise_affine=config.elementwise_affine,
                norm_eps=config.norm_eps,
                clamp_min=config.clamp_min,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx
            )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        # print(attention_mask.shape, attention_mask)
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        if torch.isnan(hidden_states).any():
            print(f"in HybridBlock ⚠️ hidden_states after layer NaN:", torch.isnan(hidden_states).any().item(), 
                  "hidden_states:", hidden_states, ", input:", residual, ", fuse_norm:", self.config.fuse_norm, ", hidden_state shape:", hidden_states.shape)
            raise ValueError(f"hidden_states NaN after layer {self.layer_idx}. hidden_states: {hidden_states}, input: {residual}")
        
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        outputs = (hidden_states, attentions, past_key_values)

        return outputs
    
class MyPreTrainedModel(PreTrainedModel):

    config_class = HybridConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['HybridBlock']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weight(
        self,
        module: nn.Module,
        prenorm_residual_strategy: Optional[str] = 'rescale',
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        
        if prenorm_residual_strategy is not None:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                if prenorm_residual_strategy == 'rescale':
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)
                elif prenorm_residual_strategy == 'zero':
                    nn.init.zeros_(p)
                else:
                    raise ValueError(f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}")

class HybridModel(MyPreTrainedModel):
    def __init__(self, config: HybridConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([HybridBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        if output_attentions:
            warnings.warn("`NSAModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(f"self.training {self.training}, use_cache: {use_cache}, return_dict: {return_dict}")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds
        # if torch.isnan(hidden_states).any():
        #     print("input_ids min/max:", input_ids.min(), input_ids.max(), "vocab_size:", self.embeddings.num_embeddings)
        #     print("embeddings weight stats:", self.embeddings.weight.min(), self.embeddings.weight.max(), self.embeddings.weight.std())
            
        if use_cache and not isinstance(past_key_values, Cache): # initialize past_key_values
            past_key_values = Cache.from_legacy_cache(past_key_values) 

        if self.gradient_checkpointing and self.training and use_cache:
            warnings.warn(
                "Gradient Checkpointing is not compatible with `use_cache=True`. Setting `use_cache=False`..."
            )
            use_cache = False
        
        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        idx = 0
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            residual = hidden_states
            if torch.isnan(hidden_states).any():
                raise ValueError(f"hidden_states NaN before layer {idx}. hidden_states: {hidden_states}, input_ids: {input_ids}")
            # print(f"in HybridModel ⚠️ hidden_states before layer {idx} NaN:", torch.isnan(hidden_states).any().item())
            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    **kwargs
                )
            else:
                hidden_states, attentions, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs
                )
            # print(f"in HybridModel ⚠️ hidden_states after layer {idx} NaN:", torch.isnan(hidden_states).any().item())
            if torch.isnan(hidden_states).any().item():
                raise ValueError(f"hidden_states NaN after layer {idx}. hidden_states: {hidden_states}, input: {residual}")
            idx += 1
            # print("done layer idx:", idx)
            if output_attentions:
                all_attns += (attentions,)
        
        # print("in HybridModel ⚠️ hidden_states before norm NaN:", torch.isnan(hidden_states).any().item())
        hidden_states = self.norm(hidden_states)
        # print("in HybridModel ⚠️ hidden_states after norm NaN:", torch.isnan(hidden_states).any().item())
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )
    
class HybridForCausalLM(MyPreTrainedModel, GenerationMixin):

    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = HybridModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings
    
    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_encoder(self):
        return self.model
    
    def tie_weights(self):
        super().tie_weights()

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception
        
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        logits_to_keep: Optional[int] = None,
        **kwargs
    ):
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:] # Keep only the last token for the incremental forward pass
        if inputs_embeds is not None and len(past_key_values) == 0:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids.contiguous()} # contiguous() to ensure that the input is stored in a new chunk of memory
        
        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
        })
        return model_inputs
    
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        hidden_states = outputs[0]
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training

        loss, logits = None, None
        if not fuse_linear_and_cross_entropy or labels is None:
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:]) # Keep only the last token for the incremental forward pass
        # if logits is not None:
        #     print("in HybridForCausalLM ⚠️ logits NaN:", torch.isnan(logits).any().item(),
        #             " +Inf:", torch.isposinf(logits).any().item(),
        #             " -Inf:", torch.isneginf(logits).any().item(), 
        #             "logits:", logits, "hidden_states NaN", torch.isnan(hidden_states).any().item(), ", hidden_states:", hidden_states)
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if fuse_linear_and_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            # full_like(labels, criterion.ignore_index) 是一个全是 ignore_index 的 tensor, 其形状是 labels[:, :1] 的形状，即一个标签的形状
            # ignore_index 是一个特殊的值，用于忽略某些特定的标签，比如padding, labels[..., 1:] 表示除了第一个标签外的所有标签
            # 这段代码的作用是：1. 去掉 `labels` 的第二维的第一个元素。2. 在第二维的末尾添加一个值为 `criterion.ignore_index` 的列。
            # 去掉序列的第一个元素（可能是特殊的起始标记）。在序列末尾添加一个特殊标记（如忽略索引），用于标记序列的结束或填充。
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:] # type(output) == Tuple, = (logits, past_key_values, hidden_states, attentions)
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
