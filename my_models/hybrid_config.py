
from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig

class HybridConfig(PretrainedConfig):

    model_type = 'glaAndNsa'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_k: int = 1,
        expand_v: int = 1,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = 8192,
        num_hidden_layers: int = 16,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = 8,
        feature_map: Optional[str] = None,
        attn_mode: str = "chunk",
        use_short_conv: bool = False,
        conv_size: int = 4,
        use_output_gate: bool = True,
        clamp_min: Optional[float] = None,
        hidden_act: str = "swish", # 相当于 silu 
        max_position_embeddings: int = 4096,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        use_gk: bool = True,
        use_gv: bool = False,
        attn: Optional[Dict] = {
            'layers': (3, 7, 11, 15),
            'num_heads': 32,
            'num_kv_heads': 2,
            'head_dim': 64,
            'qkv_bias': False,
            'block_size': 64,
            'block_counts': 16,
            'window_size': 512,
            'rope_theta': 10000.,
        },
        use_cache: bool = True,
        pad_token_id: int = 128002,
        bos_token_id: int = 128000,
        eos_token_id = [128001, 128008, 128009],
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.01,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = False, # 有 kl 散度损失时需要 logits，不能和 cross entropy 融合。
        vocab_size: int = 128256,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.feature_map = feature_map
        self.attn_mode = attn_mode
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_output_gate = use_output_gate
        self.clamp_min = clamp_min
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_gk = use_gk
        self.use_gv = use_gv
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.vocab_size = vocab_size

# for sheared Llama
# def __init__(
#         self,
#         hidden_size: int = 2048,
#         expand_k: int = 1,
#         expand_v: int = 1,
#         hidden_ratio: Optional[int] = 4,
#         intermediate_size: Optional[int] = 5504,
#         num_hidden_layers: int = 24,
#         num_heads: int = 16,
#         num_kv_heads: Optional[int] = 16,
#         feature_map: Optional[str] = None,
#         attn_mode: str = "chunk",
#         use_short_conv: bool = False,
#         conv_size: int = 4,
#         use_output_gate: bool = True,
#         clamp_min: Optional[float] = None,
#         hidden_act: str = "swish", # 相当于 silu 
#         max_position_embeddings: int = 4096,
#         elementwise_affine: Optional[bool] = True,
#         norm_eps: float = 1e-5,
#         use_gk: bool = True,
#         use_gv: bool = False,
#         attn: Optional[Dict] = {
#             'layers': (7, 15, 23),
#             'num_heads': 64,
#             'num_kv_heads': 4,
#             'head_dim': 32,
#             'qkv_bias': False,
#             'block_size': 64,
#             'block_counts': 16,
#             'window_size': 512,
#             'rope_theta': 10000.,
#         },
#         use_cache: bool = True,
#         pad_token_id: int = 0,
#         bos_token_id: int = 1,
#         eos_token_id: int = 2,
#         tie_word_embeddings: bool = False,
#         initializer_range: float = 0.01,
#         fuse_norm: bool = True,
#         fuse_swiglu: bool = True,
#         fuse_cross_entropy: bool = True,
#         vocab_size: int = 32001,
#         **kwargs
#     ):