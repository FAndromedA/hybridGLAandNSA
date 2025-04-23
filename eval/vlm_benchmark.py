import sys
import os

import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_models.Llava_config import HybridLlavaConfig
from my_models.Llava_model import HybridVisionModel
from my_models.hybrid_model import HybridForCausalLM

import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset
from pycocoevalcap.cider.cider import Cider
from accelerate.utils import set_seed

def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:.2f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.2f}Yi{suffix}'

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

import random

def build_fake_itm_dataset(match_dataset, neg_ratio=0.5):
    itm_data = []
    for sample in match_dataset:
        image = sample['image']
        captions = sample['caption']
        # 正样本
        itm_data.append({
            "image": image,
            "caption": captions[0],  # 用其中一条 reference caption
            "label": 1
        })

        # 负样本（随机选别的图片的 caption）
        # if random.random() < neg_ratio:
        neg_caption = random.choice(match_dataset)['caption'][0]  # 随机选一条 caption
        while neg_caption == captions[0]:
            neg_caption = random.choice(match_dataset)['caption'][0]
        # 确保不是同一张图片的 caption
        itm_data.append({
            "image": image,
            "caption": neg_caption,
            "label": 0
        })

    return itm_data

def do_metrics(predictions, references):
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    # BLEU-4
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    print(f"BLEU: {bleu_score['bleu']:.4f}")

    # ROUGE-L
    rouge_score = rouge_metric.compute(predictions=predictions, references=[r[0] for r in references])
    print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")

    # === pycocoevalcap CIDEr ===
    # 转成 pycocoevalcap 格式
    gts = {str(i): [cap for cap in references[i]] for i in range(len(references))}
    res = {str(i): [predictions[i]] for i in range(len(predictions))}

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    print(f"CIDEr: {cider_score:.4f}")

def evaluate_image_text_matching(model, dataset, tokenizer): # nlphuji/flickr30k
    print("Starting evaluation for image-text matching...")
    # Implement the evaluation logic for image-text matching
    dataset = build_fake_itm_dataset(dataset)
    correct = 0
    total = 0
    total_true = 1
    correct_true = 1
    total_false = 1
    correct_false = 1

    for sample in tqdm.tqdm(dataset):
        image = sample['image']
        caption = sample['caption']
        label = sample['label']  # 1: match, 0: not match
        # prompt for ep1
        # prompt = f"Image content: <image>. Text description: {caption}. Does the image content match the text description? Answer \"yes\" or \"no\"."
        messages = []
        prompt = f"Image content: <image>. Text description: \"{caption}\". Does the image content roughly match the text description? Answer yes or no."
        # prompt = f"Does the image roughly match the text description? Image: <image>. Text Description: {caption}. Answer yes or no." # output1
        # prompt = f"Given the <image>, does the following text description: \"{caption}\" roughly match the image? Answer yes or no." # output10

        messages.append({"role": "user", "content": prompt.replace('<image>', model.config.start_of_image_token + model.config.image_special_token + model.config.end_of_image_token)})

        user_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, images=[image], max_new_tokens=10)
            decoded_text = tokenizer.decode(output[0], skip_special_tokens=False)
            answer = extract_assistant_reply(decoded_text, "1")

        if ("yes" in answer.lower() and label == 1) or ("no" in answer.lower() and label == 0):
            correct += 1
            if label == 1:
                correct_true += 1
            else:
                correct_false += 1
        if label == 1:
            total_true += 1
        else:
            total_false += 1
        total += 1
        print(f"Caption: {caption}")
        print(f"acc: {correct}/{total} ({correct / total:.2%}), answer: {answer}, label: {label}, acc_true: {correct_true}/{total_true} ({correct_true / total_true:.2%}), acc_false: {correct_false}/{total_false} ({correct_false / total_false:.2%})")
    print(f"[图文匹配] Accuracy: {correct / total:.2%}")
    pass

def evaluate_vqa(model, dataset, tokenizer): # merve/vqav2-small
    # Implement the evaluation logic for VQA
    print("Starting evaluation for VQA...")
    correct = 0
    total = 0
    predictions = []
    references = []

    for sample in tqdm.tqdm(dataset):
        image = sample['image']
        question = sample['question']
        answer = sample['multiple_choice_answer']

        # prompt = f"Given the image: <image>. Please answer this question briefly: {question}"
        prompt = f"<image> {question}"
        messages = []
        messages.append({"role": "user", "content": prompt.replace('<image>', model.config.start_of_image_token + model.config.image_special_token + model.config.end_of_image_token)})

        user_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, images=[image], max_new_tokens=20)
            decoded_text = tokenizer.decode(output[0], skip_special_tokens=False)
            pred = extract_assistant_reply(decoded_text, "1")

        predictions.append(pred.strip()) 
        references.append([answer.strip()])

        if answer.lower() in pred.lower():
            correct += 1
        total += 1
        print(f"Question: {question}, answer: {answer}")
        print(f"acc: {correct}/{total} ({correct / total:.2%}), pred: {pred}")
    
    print(f"[VQA] Accuracy: {correct / total:.2%}")
    do_metrics(predictions, references)

def evaluate_captioning(model, dataset, tokenizer): # "Multimodal-Fatima/COCO_captions_test"
    # Implement the evaluation logic for image captioning
    # cider_metric = evaluate.load("cider")
    print("Starting evaluation for image captioning...")

    predictions = []
    references = []

    for sample in tqdm.tqdm(dataset):
        image = sample['image']
        refs = sample['sentences_raw']  # List[str]

        prompt = "Image: <image>. Please generate a description for this image."
        messages = []
        messages.append({"role": "user", "content": prompt.replace('<image>', model.config.start_of_image_token + model.config.image_special_token + model.config.end_of_image_token)})

        user_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, images=[image], max_new_tokens=500)
            decoded_text = tokenizer.decode(output[0], skip_special_tokens=False)
            caption = extract_assistant_reply(decoded_text, "1")

        # predictions.append(refs[0].strip() if random.random() < 0.5 else "no")  
        # 生成的描述
        predictions.append(caption.strip()) 
        references.append([r.strip() for r in refs])  # list of references per sample

    # save the predictions and references to files
    with open("predictions_captioning.txt", "w") as f:
        for pred in predictions:
            f.write(pred + "\n")

    do_metrics(predictions, references)
    

def evaluate_science_qa(model, dataset, tokenizer): # HuggingFaceM4/A-OKVQA
    # Implement the evaluation logic for ScienceQA
    print("Starting evaluation for ScienceQA...")
    correct = 0
    total = 0
    for sample in tqdm.tqdm(dataset):
        question = sample['question']
        choices = sample['choices']
        answer = sample['correct_choice_idx']
        answer = 'A' if answer == 0 else 'B' if answer == 1 else 'C' if answer == 2 else 'D' 
        if 'image' in sample:
            image = sample['image']
        else:
            image = None

        if image is not None: 
            prompt = f"<image>\nQuestion: {question}\nOptions: {', '.join(choices)}\nPlease choose the letter of the correct answer."
        else:
            prompt = f"Question: {question}\nOptions: {', '.join(choices)}\nPlease choose the letter of the correct answer."
        messages = []
        messages.append({"role": "system", "content": "Please answer the question the user asked. For example:\"Question: What is the capital of France? Choices: (A) Paris, (B) London, (C) Berlin, (D) Madrid. Please choose the letter of the only correct answer.\" because the capital city of France is Paris, you should reply with:\"(A)\""})
        messages.append({"role": "user", "content": prompt.replace('<image>', model.config.start_of_image_token + model.config.image_special_token + model.config.end_of_image_token)})

        user_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10)
            decoded_text = tokenizer.decode(output[0], skip_special_tokens=False)
            pred = extract_assistant_reply(decoded_text, "1")

        if answer.lower() in pred.lower():
            correct += 1
        total += 1
    print(f"[ScienceQA] Accuracy: {correct / total:.2%}")

def main():
    model_path = '/root/hybridGLAandNSA/ckpts_sft_llava/epoch_2' # '/root/hybridGLAandNSA/ckpts_sft_llava/epoch_0_w_dpo' # 

    dtype = torch.bfloat16
    
    test_config =  HybridLlavaConfig.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype)
    test_model = HybridVisionModel.from_pretrained(model_path, config=test_config, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    test_model.eval()
    test_model.to("cuda")

    test_num_parameters = test_model.num_parameters()
    print(f"The test model has {test_num_parameters}({sizeof_fmt(test_num_parameters)}) parameters.")
    
    set_seed(42)
    # match_dataset = load_dataset("nlphuji/flickr30k", split="test")
    vqa_dataset = load_dataset("merve/vqav2-small", split="validation")
    # caption_dataset = load_dataset("Multimodal-Fatima/COCO_captions_test", split="test")
    # scienceqa_dataset = load_dataset("HuggingFaceM4/A-OKVQA", split="test")

    # evaluate_image_text_matching(test_model, match_dataset, tokenizer)
    evaluate_vqa(test_model, vqa_dataset, tokenizer)
    # evaluate_captioning(test_model, caption_dataset, tokenizer)
    # evaluate_science_qa(test_model, scienceqa_dataset, tokenizer)

if __name__ == "__main__":
    main()