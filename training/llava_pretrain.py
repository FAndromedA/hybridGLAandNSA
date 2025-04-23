import os
import sys
import time
import math
import logging
import argparse
import warnings
from datetime import timedelta

import torch
from torch import nn, optim
import torch.nn.functional as F

import torch.distributed as dist
from contextlib import nullcontext
from accelerate import Accelerator
from accelerate.logging import get_logger
logger = get_logger(__name__)

from accelerate.utils import set_seed, InitProcessGroupKwargs
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_models.hybrid_model import HybridForCausalLM
from my_models.Llava_model import HybridVisionModel
from my_models.Llava_config import HybridLlavaConfig
from dataset import VLMDataset


def init_model(model_config: HybridLlavaConfig):
    model = HybridVisionModel(model_config)
    text_model_name_or_path = "/root/hybridGLAandNSA/ckpts_train_dpo" # "/root/hybridGLAandNSA/ckpts_train_sft/checkpoint-96587"
    model.text_model = AutoModelForCausalLM.from_pretrained(text_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(text_model_name_or_path)
    model.train()
    # freeze all parameters except the connector
    for name, param in model.named_parameters():
        if "mlp_connector" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # if hasattr(model.text_model.model, "layers"):
    #     last_two_layers = model.text_model.model.layers[-2:]
    #     for layer in last_two_layers:
    #         for param in layer.parameters():
    #             param.requires_grad = True

    for name, param in model.vision_model.named_parameters():
        if "head" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
        print(f"{name}: {param.requires_grad}")
    # exit(0)
    return model, tokenizer

def analyze_model(model, model_name):
    def sizeof_fmt(num, suffix='B'):
        for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
            if abs(num) < 1024.0:
                return f'{num:.2f}{unit}{suffix}'
            num /= 1024.0
        return f'{num:.2f}Yi{suffix}'

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of {model_name} total params:{total_params}({sizeof_fmt(total_params)})")
    print(f"number of {model_name} total trainable params:{total_trainable_params}({sizeof_fmt(total_trainable_params)})")

def main(args=None):

    ipg_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[ipg_handler])
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    set_seed(42)

    save_path = args.output_dir
    model, tokenizer = init_model(HybridLlavaConfig())
    
    
    if accelerator.is_main_process:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")
        print(model.config)
        analyze_model(model, "llava")
        analyze_model(model.text_model, "llava text model")
        analyze_model(model.vision_model, "llava vision model") 
    
    train_ds = VLMDataset(
        tokenizer=tokenizer,
        jsonl_path=args.data_path,
        images_path=args.images_path,
        preprocess=model.processor,
        image_size=model.config.vision_config.image_size,
        max_length=1024,
        image_special_token=model.config.image_special_token,
        start_of_image_token=model.config.start_of_image_token,
        end_of_image_token=model.config.end_of_image_token,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    max_steps = args.epochs * num_update_steps_per_epoch
    # overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_steps,
    )

    train_loader, model, optimizer, lr_scheduler = accelerator.prepare(
        train_loader, model, optimizer, lr_scheduler
    )
    
    completed_steps = 0
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    if accelerator.is_main_process:
        print(f"Start training for {args.epochs} epochs")
        print(f"Total training samples: {len(train_loader)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Warmup steps: {args.warmup_steps}")
    
    accelerator.wait_for_everyone()

    for epoch in range(args.epochs):
        
        for step, (X, Y, loss_mask, pixel_tensors) in enumerate(train_loader):
            # logger.info(f"epoch {epoch} step {step} device {accelerator.device} start")
            X = X.to(accelerator.device)
            Y = Y.to(accelerator.device)
            loss_mask = loss_mask.to(accelerator.device)
            pixel_tensors = pixel_tensors.to(accelerator.device)
            
            outputs = model(
                input_ids=X,
                pixel_values=pixel_tensors
            )
            logits = outputs.logits
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), 
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if (step > 0 and step % args.gradient_accumulation_steps == 0) or step == len(train_loader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                completed_steps += 1
                
            if accelerator.is_main_process and step % args.log_interval == 0:
                logger.info(f"epoch {epoch} step {step} / {len(train_loader)} loss: {loss:.5f} lr: {lr_scheduler.get_last_lr()[0]:.7f}") 

            if completed_steps > 0 and completed_steps % args.save_interval == 0:
                accelerator.wait_for_everyone()
                # save checkpoint
                output_dir = f"step_{completed_steps}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                # save model weight
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_dir)
                accelerator.save_state(output_dir)

            # logger.info(f"epoch {epoch} step {step} device {accelerator.device} end")
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)

    pass


import argparse    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid LLava Pretrain")
    parser.add_argument("--output_dir", type=str, default="/root/hybridGLAandNSA/ckpts_pretrain_llava")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_path", type=str, default="/root/LLaVA-CC3M-Pretrain-595K/chat.json")
    parser.add_argument("--images_path", type=str, default="/root/LLaVA-CC3M-Pretrain-595K/images")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=4000)
    parser.add_argument("--warmup_steps", type=int, default=750)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()


    main(args)