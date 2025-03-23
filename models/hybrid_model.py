
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from fla.layers.nsa import NativeSparseAttention
from fla.layers.attn import Attention

from fla.layers.gla import GatedLinearAttention
from fla.models.gla.configuration_gla import GLAConfig
from fla.models.utils import Cache
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules import GatedMLP as GLAMLP
from fla.modules import RMSNorm

class MyBlock(nn.Modules):
    def __init__(self,  config: GLAConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = NativeSparseAttention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                head_dim=64,
                qkv_bias=config.attn['qkv_bias'],
                block_size=64,
                block_counts=16,
                window_size=config.attn['window_size'],
                rope_theta=config.attn['rope_theta'],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx
            )
        else:
            self.attn = GatedLinearAttention(
                mode=config.attn_mode,
                hiddne_size=config.hidden_size,
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
        self.mlp_norm = ()