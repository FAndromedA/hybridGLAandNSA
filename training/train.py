import torch
import warnings

from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM

import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.hybrid_config import HybridConfig
from models.hybrid_model import HybridModel, HybridBlock, HybridForCausalLM

def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:.2f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.2f}Yi{suffix}'

# def convert_llama_weights(
#     llama: str,
#     config: HybridConfig,
#     output: str,
#     precision: str = "float32",
# ):
#     AutoTokenizer.from_pretrained(llama).save_pretrained(output)
#     llama = AutoModelForCausalLM.from_pretrained(llama, torch_dtype=precision)
#     print(f"Loading model {llama}")

#     config = HybridConfig.from_pretrained(llama.config)
#     model = AutoModelForCausalLM.from_config(config)
#     if precision in ["float16", "fp16"]:
#         model = model.to(dtype=torch.float16)
#     if precision in ["bfloat16", "bf16"]:
#         model = model.to(dtype=torch.bfloat16)
#     num_parameters = model.num_parameters()
#     print(f"Initializing the model from the config:\n{config}\n{model}")
#     print(f"Number of parameters: {num_parameters} ({sizeof_fmt(num_parameters)})")

#     print("Copying the weights from Llama to the model ...")
#     vocab_size = llama.model.embed_tokens.weight.shape[0]
#     if model.model.embeddings.weight.shape[0] != vocab_size:
#         warnings.warn(f"Llama and the model have different embedding sizes "
#                       f"({vocab_size} vs {model.model.embeddings.weight.shape[0]}), "
#                       f"the model embeddings will be extended with randomly initialized values or truncated")
#         vocab_size = min(model.model.embeddings.weight.shape[0], vocab_size)
#     print("llama.model.embed_tokens                        -> model.model.embeddings")
    
#     model.model.embeddings.weight.data[:vocab_size].copy_(llama.model.embed_tokens.weight[:vocab_size])
#     torch.testing.assert_close(model.model.embeddings.weight[:vocab_size], llama.model.embed_tokens.weight[:vocab_size])
    
#     for i in range(config.num_hidden_layers):
#         if hasattr(model.model.layers[i], 'attn_norm'):
#             if model.model.layers[i].attn_norm.weight is not None:
#                 print(f"llama.model.layers.{i}.input_layernorm.weight -> model.model.layers.{i}.attn_norm.weight")
#                 model.model.layers[i].attn_norm.weight.data.copy_(llama.model.layers[i].input_layernorm.weight)
#                 torch.testing.assert_close(model.model.layers[i].attn_norm.weight, 
#                                         llama.model.layers[i].input_layernorm.weight)
#             if model.model.layers[i].attn_norm.bias is not None:
#                 print(f"llama.model.layers{i}.input_layernorm.bias -> model.model.layers{i}.attn_norm.bias")
#                 model.model.layers[i].attn_norm.bias.data.copy_(llama.model.layers[i].input_layernorm.bias)
#                 torch.testing.assert_close(model.model.layers[i].attn_norm.bias,
#                                            llama.model.layers[i].input_layernorm.bias)
#             model.model.layers[i].attn_norm.eps = llama.model.layers[i].input_layernorm.variance_epsilon
        
#         print(f"llama.model.layers{i}.attn.q_proj.weight -> model.model.layers{i}.attn.q_proj.weight")
#         model.model.layers[i].attn.q_proj.weight.data.copy_(llama.model.layers[i].self_attn.q_proj.weight)
#         torch.testing.assert_close(model.model.layers[i].attn.q_proj.weight, llama.model.layers[i].self_attn.q_proj.weight)
#         if hasattr(llama.model.layers[i].self_attn.q_proj, 'bias') and hasattr(model.model.layers[i].attn.q_proj, 'bias'):
#             print(f"llama.model.layers{i}.attn.q_proj.bias -> model.model.layers{i}.attn.q_proj.bias")
#             model.model.layers[i].attn.q_proj.bias.data.copy_(llama.model.layers[i].self_attn.q_proj.bias)
#             torch.testing.assert_close(model.model.layers[i].attn.q_proj.bias, llama.model.layers[i].self_attn.q_proj.bias)

#         print(f"llama.model.layers{i}.attn.k_proj.weight -> model.model.layers{i}.attn.k_proj.weight")
#         model.model.layers[i].attn.k_proj.weight.data.copy_(llama.model.layers[i].self_attn.k_proj.weight)
#         torch.testing.assert_close(model.model.layers[i].attn.k_proj.weight, llama.model.layers[i].self_attn.k_proj.weight)
#         if hasattr(llama.model.layers[i].self_attn.k_proj, 'bias') and hasattr(model.model.layers[i].attn.k_proj, 'bias'):
#             print(f"llama.model.layers{i}.attn.k_proj.bias -> model.model.layers{i}.attn.k_proj.bias")
#             model.model.layers[i].attn.k_proj.bias.data.copy_(llama.model.layers[i].self_attn.k_proj.bias)
#             torch.testing.assert_close(model.model.layers[i].attn.k_proj.bias, llama.model.layers[i].self_attn.k_proj.bias)

#         print(f"llama.model.layers{i}.attn.v_proj.weight -> model.model.layers{i}.attn.v_proj.weight")
#         model.model.layers[i].attn.v_proj.weight.data.copy_(llama.model.layers[i].self_attn.v_proj.weight)
#         torch.testing.assert_close(model.model.layers[i].attn.v_proj.weight, llama.model.layers[i].self_attn.v_proj.weight)
#         if hasattr(llama.model.layers[i].self_attn.v_proj, 'bias') and hasattr(model.model.layers[i].attn.v_proj, 'bias'):
#             print(f"llama.model.layers{i}.attn.v_proj.bias -> model.model.layers{i}.attn.v_proj.bias")
#             model.model.layers[i].attn.v_proj.bias.data.copy_(llama.model.layers[i].self_attn.v_proj.bias)
#             torch.testing.assert_close(model.model.layers[i].attn.v_proj.bias, llama.model.layers[i].self_attn.v_proj.bias)

#         print(f"llama.model.layers{i}.attn.o_proj.weight -> model.model.layers{i}.attn.o_proj.weight")
#         model.model.layers[i].attn.o_proj.weight.data.copy_(llama.model.layers[i].self_attn.o_proj.weight)
#         torch.testing.assert_close(model.model.layers[i].attn.o_proj.weight, llama.model.layers[i].self_attn.o_proj.weight)
#         if hasattr(llama.model.layers[i].self_attn.o_proj, 'bias') and hasattr(model.model.layers[i].attn.o_proj, 'bias'):
#             print(f"llama.model.layers{i}.attn.o_proj.bias -> model.model.layers{i}.attn.o_proj.bias")
#             model.model.layers[i].attn.o_proj.bias.data.copy_(llama.model.layers[i].self_attn.o_proj.bias)
#             torch.testing.assert_close(model.model.layers[i].attn.o_proj.bias, llama.model.layers[i].self_attn.o_proj.bias)

#         if hasattr(model.model.layers[i], 'mlp_norm'):
#             if model.model.layers[i].mlp_norm.weight is not None:
#                 print(f"llama.model.layers{i}.post_attention_layernorm.weight -> model.model.layers{i}.mlp_norm.weight")
#                 model.model.layers[i].mlp_norm.weight.data.copy_(llama.model.layers[i].post_attention_layernorm.weight)
#                 torch.testing.assert_close(model.model.layers[i].mlp_norm.weight,
#                                            llama.model.layers[i].post_attention_layernorm.weight)
            
#             if model.model.layers[i].mlp_norm.bias is not None:
#                 print(f"llama.model.layers{i}.post_attention_layernorm.bias -> model.model.layers{i}.mlp_norm.bias")
#                 model.model.layers[i].mlp_norm.bias.data.copy_(llama.model.layers[i].post_attention_layernorm.bias)
#                 torch.testing.assert_close(model.model.layers[i].mlp_norm.bias,
#                                            llama.model.layers[i].post_attention_layernorm.bias)
        
#         print(f"llama.model.layers.{i}.mlp.gate_proj.weight -> model.model.layers.{i}.mlp.gate_proj.weight")
#         model.model.layers[i].mlp.gate_proj.weight.data.copy_(llama.model.layers[i].mlp.gate_proj.weight)
#         torch.testing.assert_close(model.model.layers[i].mlp.gate_proj.weight, llama.model.layers[i].mlp.gate_proj.weight)
#         print(f"llama.model.layers.{i}.mlp.up_proj.weight -> model.model.layers.{i}.mlp.up_proj.weight")
#         model.model.layers[i].mlp.up_proj.weight.data.copy_(llama.model.layers[i].mlp.up_proj.weight)
#         torch.testing.assert_close(model.model.layers[i].mlp.up_proj.weight, llama.model.layers[i].mlp.up_proj.weight)

#         print(f"llama.model.layers.{i}.mlp.down_proj.weight -> model.model.layers.{i}.mlp.down_proj.weight")
#         model.model.layers[i].mlp.down_proj.weight.data.copy_(llama.model.layers[i].mlp.down_proj.weight)
#         torch.testing.assert_close(model.model.layers[i].mlp.down_proj.weight,
#                                    llama.model.layers[i].mlp.down_proj.weight)
    
#     if model.model.norm.weight is not None:
#         print("llama.model.norm.weight -> model.model.norm.weight")
#         model.model.norm.weight.data.copy_(llama.model.norm.weight)
#         torch.testing.assert_close(model.model.norm.weight, llama.model.norm.weight)
#     if model.model.norm.bias is not None:
#         print("llama.model.norm.bias -> model.model.norm.bias")
#         model.model.norm.bias.data.copy_(llama.model.norm.bias)
#         torch.testing.assert_close(model.model.norm.bias, llama.model.norm.bias)
#     model.model.norm.eps = llama.model.norm.variance_epsilon

#     if not model.config.tie_word_embeddings:
#         print("llama.model.lm_head.weight -> model.lm_head.weight")
#         model.lm_head.weight.data[:vocab_size].copy_(llama.lm_head.weight[:vocab_size])
#         torch.testing.assert_close(model.lm_head.weight[:vocab_size], llama.lm_head.weight[:vocab_size])
#     model.config.rope_theta = llama.config.rope_theta

#     print(f"Saving converted model to {output} ...\n{model}")
#     model.save_pretrained(output)

def convert_llama_weights(
    llama: AutoModelForCausalLM,
    config: HybridConfig,
    model: HybridForCausalLM,
    output: str,
    precision: str = "float32",
):
    # AutoTokenizer.from_pretrained(llama).save_pretrained(output)
    # llama = AutoModelForCausalLM.from_pretrained(llama, torch_dtype=precision)
    # print(f"Teacher model {llama}")

    # config = HybridConfig.from_pretrained(llama.config)
    # print(f"Student config {config}")

    # model = AutoModelForCausalLM.from_config(config)
    if precision in ["float16", "fp16"]:
        model = model.to(dtype=torch.float16)
    if precision in ["bfloat16", "bf16"]:
        model = model.to(dtype=torch.bfloat16)
    # num_parameters = model.num_parameters()
    # print(f"Initializing the model from the config:\n{config}\n{model}")
    # print(f"Number of parameters: {num_parameters} ({sizeof_fmt(num_parameters)})")

    print("Copying the weights from Llama to the model ...")
    vocab_size = llama.model.embed_tokens.weight.shape[0]
    if model.model.embeddings.weight.shape[0] != vocab_size:
        warnings.warn(f"Llama and the model have different embedding sizes "
                      f"({vocab_size} vs {model.model.embeddings.weight.shape[0]}), "
                      f"the model embeddings will be extended with randomly initialized values or truncated")
        vocab_size = min(model.model.embeddings.weight.shape[0], vocab_size)
    print("llama.model.embed_tokens                        -> model.model.embeddings")
    
    model.model.embeddings.weight.data[:vocab_size].copy_(llama.model.embed_tokens.weight[:vocab_size])
    torch.testing.assert_close(model.model.embeddings.weight[:vocab_size], llama.model.embed_tokens.weight[:vocab_size])
    
    for i in range(config.num_hidden_layers):
        if hasattr(model.model.layers[i], 'attn_norm'):
            if model.model.layers[i].attn_norm.weight is not None:
                print(f"llama.model.layers.{i}.input_layernorm.weight -> model.model.layers.{i}.attn_norm.weight")
                model.model.layers[i].attn_norm.weight.data.copy_(llama.model.layers[i].input_layernorm.weight)
                torch.testing.assert_close(model.model.layers[i].attn_norm.weight, 
                                        llama.model.layers[i].input_layernorm.weight)
            if model.model.layers[i].attn_norm.bias is not None:
                print(f"llama.model.layers{i}.input_layernorm.bias -> model.model.layers{i}.attn_norm.bias")
                model.model.layers[i].attn_norm.bias.data.copy_(llama.model.layers[i].input_layernorm.bias)
                torch.testing.assert_close(model.model.layers[i].attn_norm.bias,
                                           llama.model.layers[i].input_layernorm.bias)
            model.model.layers[i].attn_norm.eps = llama.model.layers[i].input_layernorm.variance_epsilon
        
        print(f"llama.model.layers{i}.attn.q_proj.weight -> model.model.layers{i}.attn.q_proj.weight")
        model.model.layers[i].attn.q_proj.weight.data.copy_(llama.model.layers[i].self_attn.q_proj.weight)
        torch.testing.assert_close(model.model.layers[i].attn.q_proj.weight, llama.model.layers[i].self_attn.q_proj.weight)
        if hasattr(llama.model.layers[i].self_attn.q_proj, 'bias') and hasattr(model.model.layers[i].attn.q_proj, 'bias'):
            if model.model.layers[i].attn.q_proj.bias is not None:
                print(f"llama.model.layers{i}.attn.q_proj.bias -> model.model.layers{i}.attn.q_proj.bias")
                model.model.layers[i].attn.q_proj.bias.data.copy_(llama.model.layers[i].self_attn.q_proj.bias)
                torch.testing.assert_close(model.model.layers[i].attn.q_proj.bias, llama.model.layers[i].self_attn.q_proj.bias)

        print(f"llama.model.layers{i}.attn.k_proj.weight -> model.model.layers{i}.attn.k_proj.weight")
        model.model.layers[i].attn.k_proj.weight.data.copy_(llama.model.layers[i].self_attn.k_proj.weight)
        torch.testing.assert_close(model.model.layers[i].attn.k_proj.weight, llama.model.layers[i].self_attn.k_proj.weight)
        if hasattr(llama.model.layers[i].self_attn.k_proj, 'bias') and hasattr(model.model.layers[i].attn.k_proj, 'bias'):
            if model.model.layers[i].attn.k_proj.bias is not None:
                print(f"llama.model.layers{i}.attn.k_proj.bias -> model.model.layers{i}.attn.k_proj.bias")
                model.model.layers[i].attn.k_proj.bias.data.copy_(llama.model.layers[i].self_attn.k_proj.bias)
                torch.testing.assert_close(model.model.layers[i].attn.k_proj.bias, llama.model.layers[i].self_attn.k_proj.bias)

        print(f"llama.model.layers{i}.attn.v_proj.weight -> model.model.layers{i}.attn.v_proj.weight")
        model.model.layers[i].attn.v_proj.weight.data.copy_(llama.model.layers[i].self_attn.v_proj.weight)
        torch.testing.assert_close(model.model.layers[i].attn.v_proj.weight, llama.model.layers[i].self_attn.v_proj.weight)
        if hasattr(llama.model.layers[i].self_attn.v_proj, 'bias') and hasattr(model.model.layers[i].attn.v_proj, 'bias'):
            if model.model.layers[i].attn.v_proj.bias is not None:
                print(f"llama.model.layers{i}.attn.v_proj.bias -> model.model.layers{i}.attn.v_proj.bias")
                model.model.layers[i].attn.v_proj.bias.data.copy_(llama.model.layers[i].self_attn.v_proj.bias)
                torch.testing.assert_close(model.model.layers[i].attn.v_proj.bias, llama.model.layers[i].self_attn.v_proj.bias)

        print(f"llama.model.layers{i}.attn.o_proj.weight -> model.model.layers{i}.attn.o_proj.weight")
        model.model.layers[i].attn.o_proj.weight.data.copy_(llama.model.layers[i].self_attn.o_proj.weight)
        torch.testing.assert_close(model.model.layers[i].attn.o_proj.weight, llama.model.layers[i].self_attn.o_proj.weight)
        if hasattr(llama.model.layers[i].self_attn.o_proj, 'bias') and hasattr(model.model.layers[i].attn.o_proj, 'bias'):
            if model.model.layers[i].attn.o_proj.bias is not None:
                print(f"llama.model.layers{i}.attn.o_proj.bias -> model.model.layers{i}.attn.o_proj.bias")
                model.model.layers[i].attn.o_proj.bias.data.copy_(llama.model.layers[i].self_attn.o_proj.bias)
                torch.testing.assert_close(model.model.layers[i].attn.o_proj.bias, llama.model.layers[i].self_attn.o_proj.bias)

        if hasattr(model.model.layers[i], 'mlp_norm'):
            if model.model.layers[i].mlp_norm.weight is not None:
                print(f"llama.model.layers{i}.post_attention_layernorm.weight -> model.model.layers{i}.mlp_norm.weight")
                model.model.layers[i].mlp_norm.weight.data.copy_(llama.model.layers[i].post_attention_layernorm.weight)
                torch.testing.assert_close(model.model.layers[i].mlp_norm.weight,
                                           llama.model.layers[i].post_attention_layernorm.weight)
            
            if model.model.layers[i].mlp_norm.bias is not None:
                print(f"llama.model.layers{i}.post_attention_layernorm.bias -> model.model.layers{i}.mlp_norm.bias")
                model.model.layers[i].mlp_norm.bias.data.copy_(llama.model.layers[i].post_attention_layernorm.bias)
                torch.testing.assert_close(model.model.layers[i].mlp_norm.bias,
                                           llama.model.layers[i].post_attention_layernorm.bias)
        
        print(f"llama.model.layers.{i}.mlp.gate_proj.weight -> model.model.layers.{i}.mlp.gate_proj.weight")
        model.model.layers[i].mlp.gate_proj.weight.data.copy_(llama.model.layers[i].mlp.gate_proj.weight)
        torch.testing.assert_close(model.model.layers[i].mlp.gate_proj.weight, llama.model.layers[i].mlp.gate_proj.weight)
        print(f"llama.model.layers.{i}.mlp.up_proj.weight -> model.model.layers.{i}.mlp.up_proj.weight")
        model.model.layers[i].mlp.up_proj.weight.data.copy_(llama.model.layers[i].mlp.up_proj.weight)
        torch.testing.assert_close(model.model.layers[i].mlp.up_proj.weight, llama.model.layers[i].mlp.up_proj.weight)

        print(f"llama.model.layers.{i}.mlp.down_proj.weight -> model.model.layers.{i}.mlp.down_proj.weight")
        model.model.layers[i].mlp.down_proj.weight.data.copy_(llama.model.layers[i].mlp.down_proj.weight)
        torch.testing.assert_close(model.model.layers[i].mlp.down_proj.weight,
                                   llama.model.layers[i].mlp.down_proj.weight)
    
    if model.model.norm.weight is not None:
        print("llama.model.norm.weight -> model.model.norm.weight")
        model.model.norm.weight.data.copy_(llama.model.norm.weight)
        torch.testing.assert_close(model.model.norm.weight, llama.model.norm.weight)
    if model.model.norm.bias is not None:
        print("llama.model.norm.bias -> model.model.norm.bias")
        model.model.norm.bias.data.copy_(llama.model.norm.bias)
        torch.testing.assert_close(model.model.norm.bias, llama.model.norm.bias)
    model.model.norm.eps = llama.model.norm.variance_epsilon

    if not model.config.tie_word_embeddings:
        print("llama.model.lm_head.weight -> model.lm_head.weight")
        model.lm_head.weight.data[:vocab_size].copy_(llama.lm_head.weight[:vocab_size])
        torch.testing.assert_close(model.lm_head.weight[:vocab_size], llama.lm_head.weight[:vocab_size])
    model.config.rope_theta = llama.config.rope_theta

    print(f"Saving converted model to {output} ...\n{model}")
    model.save_pretrained(output)

def test_load():
    config = AutoConfig.from_pretrained("/root/Sheared-LLaMA-1.3B", local_files_only=True)
    model = AutoModel.from_pretrained("/root/Sheared-LLaMA-1.3B", config=config, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("/root/Sheared-LLaMA-1.3B")
    num_parameters = model.num_parameters()
    print(f"Initializing the model from the config:\n{config}\n{model}")
    print(f"Number of parameters: {num_parameters} ({sizeof_fmt(num_parameters)})")

# test_load()

def main():
    teacher_config =  AutoConfig.from_pretrained("/root/Sheared-LLaMA-1.3B", local_files_only=True)
    teacher_model = AutoModelForCausalLM.from_pretrained("/root/Sheared-LLaMA-1.3B", config=teacher_config, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("/root/Sheared-LLaMA-1.3B")
    teacher_num_parameters = teacher_model.num_parameters()
    print(f"Teacher model config:\n{teacher_config}\n teacher model:\n{teacher_model}")
    print(f"Number of teacher parameters: {teacher_num_parameters} ({sizeof_fmt(teacher_num_parameters)})")

    save_path = '/root/hybridGlaAndNsa/ckpts'
    tokenizer.save_pretrained(save_path)
    student_config = HybridConfig()
    student_model = HybridForCausalLM._from_config(student_config)
    student_num_parameters = student_model.num_parameters()
    print(f"Student model config:\n{student_config}\n student model:\n{student_model}")
    print(f"Number of student parameters: {student_num_parameters} ({sizeof_fmt(student_num_parameters)})")

    convert_llama_weights(teacher_model, student_config, student_model, save_path)

    pass

if __name__  == '__main__':
    main()