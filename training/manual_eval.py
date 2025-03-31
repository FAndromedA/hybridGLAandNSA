import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.hybrid_config import HybridConfig
from models.hybrid_model import HybridModel, HybridBlock, HybridForCausalLM

import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM

import torch

if __name__ == "__main__":

    model_path = "/root/hybridGlaAndNsa/ckpts1" #"/root/Llama-3.2-1B-Instruct"#  "/root/Sheared-LLaMA-1.3B-ShareGPT"
    dtype = torch.bfloat16
    test_config =  AutoConfig.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype)
    test_model = AutoModelForCausalLM.from_pretrained(model_path, config=test_config, torch_dtype=dtype, local_files_only=True)
    test_model.eval()
    test_model.to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_num_parameters = test_model.num_parameters()
    print(f"The test model has {test_num_parameters} parameters.")

    while True:
        user_input = input("Please enter your input: ")
        if user_input.lower() == "exit":
            print("Exiting the program.")
            break    
        # input_with_template = f"You are a helpful assistant. Write a response that appropriately completes the request.\n\n### Input:\n{user_input}\n\n### Response:"
        with torch.no_grad():
            inputs = tokenizer(user_input, return_tensors="pt").to("cuda:0")
            outputs = test_model.generate(inputs.input_ids, max_length=1024, do_sample=True)
            print(f"The test model generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    
    # user_input = "Hello, can you tell me a joke?"
