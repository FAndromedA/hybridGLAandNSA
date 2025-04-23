import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_models.Llava_config import HybridLlavaConfig
from my_models.Llava_model import HybridVisionModel
from my_models.hybrid_model import HybridForCausalLM

import random
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

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

if __name__ == '__main__':

    model_path = '/root/hybridGLAandNSA/ckpts_sft_llava/epoch_2'
    dtype = torch.bfloat16
    
    test_config =  HybridLlavaConfig.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype)
    test_model = HybridVisionModel.from_pretrained(model_path, config=test_config, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    test_model.eval()
    test_model.to("cuda:0")

    # for name, param in test_model.named_parameters():
    #    print(f"{name}: {param.requires_grad}")
    # test_num_parameters = test_model.num_parameters()
    # print(f"The test model has {test_num_parameters} parameters.")
    # exit(0)
    use_template = "1"

    with torch.no_grad():
        while True:
            print("============================================") 
            user_input = input("Please enter your input: ") # Provide a detail description of the given image
            # Please give me a story based on the given image before
            # Question: Which of these states is farthest north? Please choose the letter of the Only correct Choice: (A) West Virginia, (B) Louisiana, (C) Arizona, (D) Oklahoma. 
            # img_path = '/root/hybridGLAandNSA/eval/images/test_image2.jpg'
            images = None
            img_path = input("Please enter the image path: ")
            pixel_values = None
            if user_input.lower() == "exit":
                print("Exiting the program.")
                break
            if img_path.lower() == "none":
                messages = [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "system", "content": "Please answer the question the user asked. For example:\"Question: What is the capital of France? Choices: (A) Paris, (B) London, (C) Berlin, (D) Madrid. Please choose the letter of the only correct answer.\" because the capital city of France is Paris, you should reply with:\"(A)\""},
                    {"role": "user", "content": user_input},
                ]
            else: 
                if os.path.exists(img_path) == False:
                    print(f"The image path {img_path} does not exist.")
                    continue

                images = []
                user_input = test_model.config.start_of_image_token + test_model.config.image_special_token + test_model.config.end_of_image_token + user_input
                messages = [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "system", "content": "Please answer the question the user asked. For example:\"Question: What is the capital of France? Choices: (A) Paris, (B) London, (C) Berlin, (D) Madrid. Please choose the letter of the only correct answer.\" because the capital city of France is Paris, you should reply with:\"(A)\""},
                    {"role": "user", "content": user_input},
                ]
                image = Image.open(img_path)
                images.append(image)
                # pixel_values = test_model.image2tensor(image)

            user_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            
            # print(f"The input to the model is: {repr(user_input)}")
            inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
            # print(f"The input token to the model is: {inputs}")
            start_time = time.time()
            outputs = test_model.generate(inputs.input_ids, images=images, max_length=4096, do_sample=True)
            end_time = time.time()

            decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            # print(decoded_text)
            assistant_reply = extract_assistant_reply(decoded_text, use_template)

            print(f"Time taken for generation: {end_time - start_time:.2f} seconds, \
                  speed: {(outputs[0].shape[0] - inputs.input_ids.shape[1]) / (end_time - start_time):.2f} tokens/sec")
            print(f"The test model generated: {assistant_reply}")

    
    print("Test model inference completed.")
        


