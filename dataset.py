import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling

class TextDataset(Dataset):
    def __init__(self, training_data, training_label, pad_token_id):
        self.training_data = training_data
        self.training_label = training_label
        self.pad_token_id = pad_token_id
        assert self.training_data.shape == self.training_label.shape
        
    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        input_ids = self.training_data[idx]
        # 
        return {"input_ids": input_ids, "labels": self.training_label[idx], "attention_mask": input_ids.ne(self.pad_token_id)}

import json
from PIL import Image
from models.Llava_model import HybridVisionModel
from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 1024 * 1024 * 10 # 10MB

class VLMDataset(Dataset):
    def __init__(self, tokenizer, jsonl_path=None, images_path=None, hf_dataset=None, preprocess=None,
                 image_size=224, max_length=1024, image_special_token='<|reserved_special_token_1|>' * 256,
                 start_of_image_token='<|reserved_special_token_2|>', end_of_image_token='<|reserved_special_token_3|>'):
        super().__init__()
        
        assert hf_dataset is not None or (jsonl_path and images_path), \
            "Either hf_dataset or (jsonl_path + images_path) must be provided."

        if hf_dataset is not None:
            self.samples = hf_dataset
            self.from_hf = True
        else:
            self.samples = self.load_data(jsonl_path)
            self.images_path = images_path

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_len_meet = 0
        self.preprocess = preprocess
        self.image_size = image_size
        self.image_special_token = image_special_token
        self.start_of_image_token = start_of_image_token
        self.end_of_image_token = end_of_image_token
        self.bos_id = tokenizer('<|start_header_id|>assistant<|end_header_id|>\n\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|eot_id|>', add_special_tokens=False).input_ids
    
    def __len__(self):
        return len(self.samples)
    
    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            # read json
            samples = json.load(f) 
            # read jsonl
            # for _, line in enumerate(f, 1):
            #     data = json.loads(line.strip())
            #     samples.append(data)
        return samples

    def _create_chat_template(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            if self.from_hf:
                messages.append({'role': role, "content": turn['content'].replace('<image>', self.start_of_image_token + self.image_special_token + self.end_of_image_token)})
            else:
                messages.append({'role': role, "content": turn['value'].replace('<image>', self.start_of_image_token + self.image_special_token + self.end_of_image_token)})
            # 生成 image 占位符
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids) 
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask # loss_mask represents the positions of the tokens that should be used for loss calculation

    def __getitem__(self, index: int):
        sample = self.samples[index]
        
        conversations = sample["messages"] if self.from_hf else sample["conversations"]
        prompt = self._create_chat_template(conversations)
        self.max_len_meet = max(self.max_len_meet, len(prompt)) # used for debugging
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        # X, Y is the input and target for the model, and the Y is shifted by one position
        # to the right, so that the model can predict the next token
        # loss_mask is used to mask the padding tokens in the input_ids
        # and the loss will not be calculated for the padding tokens
        X = torch.tensor(input_ids[:-1], dtype=torch.long) 
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        image_list = []
        image_tensors = []

        if self.from_hf:
            images = sample["images"] # typically list of PIL.Image
            if images == None: # cause the dataset we use is mixed by multi-modal data and pure text data
                return X, Y, loss_mask, torch.zeros(1, 1, 3, self.image_size, self.image_size).to(X.device)
            if isinstance(images, list):
                image_list = images
            else:
                image_list = [images]
        else:
            image_paths = sample["image"]
            for image_name in sample["image"].split(','):
                image_path = f"{self.images_path}/{image_name.strip()}"
                image_list.append(Image.open(image_path))

        for image in image_list:
            if image.mode in ['RGBA', 'LA']:
                image = image.convert("RGB")
            if self.preprocess:
                image_tensors.append(
                    self.preprocess(
                        images=image,
                        return_tensors="pt",
                        do_resize=True,
                        size={
                            "height": self.image_size, #224
                            "width": self.image_size, #224
                        },
                    )['pixel_values'])
            else:
                raise ValueError("Preprocess function is not defined.")
        
        image_tensors = torch.stack(image_tensors, dim=0) # dim0 is picture number
        return X, Y, loss_mask, image_tensors

if __name__ == "__main__":

    from transformers import AutoModelForCausalLM
    from tqdm import tqdm
    
    training_data = torch.load(f'ultrachat_input_ids.pt', map_location="cpu")
    training_label = torch.load(f'ultrachat_labels.pt', map_location="cpu")
    
    dataset = TextDataset(training_data, training_label, pad_token_id=2)
    dataloader = DataLoader(dataset, batch_size=4)

    model_name = "HuggingFaceH4/zephyr-7b-beta"
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(dtype).cuda()

    with torch.no_grad(): 
        for j, inputs in tqdm(enumerate(dataloader)):
            input_ids = inputs["input_ids"].cuda()
            labels = inputs["labels"].cuda()
            attention_mask = inputs["attention_mask"].cuda()
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            print(output.loss)