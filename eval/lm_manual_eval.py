import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_models.hybrid_config import HybridConfig
from my_models.hybrid_model import HybridModel, HybridBlock, HybridForCausalLM

import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM

import torch

def extract_assistant_reply(output_text, use_template):
    if use_template == "0":
        return output_text
    start_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    end_token = "<|eot_id|>"
    
    if start_token in output_text:
        response = output_text.split(start_token)[-1]  # 取 assistant 后面的内容
        response = response.split(end_token)[0] if end_token in response else response  # 去掉 <|eot_id|>
        return response.strip()
    else:
        return "Assistant 回复未找到"

if __name__ == "__main__":
    # "/root/hybridGLAandNSA/ckpts_train_hybrid"
    model_path = "/root/hybridGLAandNSA/ckpts_train_dpo/checkpoint-33342"
    #"/root/hybridGLAandNSA/ckpts_train_sft/checkpoint-96587" 
    # "/root/Llama-3.2-1B-Instruct"# "/root/Sheared-LLaMA-1.3B-ShareGPT"
    dtype = torch.bfloat16
    test_config =  AutoConfig.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype)
    test_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True)
    test_model.eval()
    test_model.to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_num_parameters = test_model.num_parameters()
    for name, param in test_model.named_parameters():
       param.requires_grad = False
       # print(f"{name}: {param.requires_grad}")
    print(f"The test model has {test_num_parameters} parameters.")

    use_template = input("Do you want to use the template? (1 for yes/0 for no): ")
    while True:
        print("============================================")
        user_input = input("Please enter your input: ")
        if user_input.lower() == "exit":
            print("Exiting the program.")
            break    
        # input_with_template = f"You are a helpful assistant. Write a response that appropriately completes the request.\n\n### Input:\n{user_input}\n\n### Response:"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            # {"role": "system", "content": "Please answer the question the user asked. For example:\"Question: What is the capital of France? Choices: (A) Paris, (B) London, (C) Berlin, (D) Madrid. Please choose the letter of the only correct answer.\" because the capital city of France is Paris, you should reply with:\"(A)\""},
            {"role": "user", "content": user_input},
        ]
        with torch.no_grad():
            if use_template == "1":
                user_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"The input to the model is: {repr(user_input)}")
            # an example of template:
            #<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 07 Apr 2025\n\nYou are a helpful assistant named Nova, created by ZJH.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ntell me a story<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
            
            inputs = tokenizer(user_input, return_tensors="pt").to("cuda:0")
            # print(f"The input tensor is: {inputs.input_ids}, shape: {inputs.input_ids.shape}")
            start_time = time.time()
            outputs = test_model.generate(inputs.input_ids, max_length=1024, do_sample=True)
            end_time = time.time()

            decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            # print(decoded_text)
            assistant_reply = extract_assistant_reply(decoded_text, use_template)
            print(f"Time taken for generation: {end_time - start_time:.2f} seconds, \
                  speed: {(outputs[0].shape[0] - inputs.input_ids.shape[1]) / (end_time - start_time):.2f} tokens/sec")
            print(f"The test model generated: {assistant_reply}")
    
    # user_input = "Hello, can you tell me a joke?"
