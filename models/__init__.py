from transformers import AutoConfig, AutoModel, AutoModelForCausalLM


from .hybrid_model import HybridModel, HybridForCausalLM, HybridBlock
from .hybrid_config import HybridConfig

AutoConfig.register(HybridConfig.model_type, HybridConfig)
AutoModel.register(HybridConfig, HybridModel)
AutoModelForCausalLM.register(HybridConfig, HybridForCausalLM)

__all__ = [ "HybridModel", "HybridForCausalLM", "HybridConfig", "HybridBlock" ]