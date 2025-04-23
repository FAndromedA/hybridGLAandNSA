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
from datasets import load_from_disk

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



def main(args):
    ipg_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[ipg_handler])
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    set_seed(42)

    save_path = args.output_dir
    dtype = torch.bfloat16
    model_path = '/root/hybridGLAandNSA/ckpts_pretrain_llava/epoch_2'
    config =  HybridLlavaConfig.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype)
    model = HybridVisionModel.from_pretrained(model_path, config=config, torch_dtype=dtype).train()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.model_max_length = 3072

    for name, param in model.named_parameters():
        if "vision_model" not in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    for name, param in model.vision_model.named_parameters():
        if "head" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    if accelerator.is_main_process:
        print(model.config)
        analyze_model(model, "llava")
        analyze_model(model.text_model, "llava text model")
        analyze_model(model.vision_model, "llava vision model")

    ds_mine = load_from_disk(args.hfdataset_path)

    train_ds = VLMDataset(
        tokenizer=tokenizer,
        hf_dataset=ds_mine,
        preprocess=model.processor,
        image_size=model.config.vision_config.image_size,
        max_length=3072,
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
    
    for epoch in range(args.epochs):
        # print(f"epoch {epoch} device {accelerator.device} start")
        for step, (X, Y, loss_mask, pixel_tensors) in enumerate(train_loader):
            # print(f"epoch {epoch} step {step} device {accelerator.device} start")
            X = X.to(accelerator.device)
            Y = Y.to(accelerator.device)
            loss_mask = loss_mask.to(accelerator.device)
            pixel_tensors = pixel_tensors.to(accelerator.device)
            
            # print(f"epoch {epoch} step {step} device {accelerator.device} get outputs start")
            outputs = model(
                input_ids=X,
                pixel_values=pixel_tensors
            )
            # print(f"epoch {epoch} step {step} device {accelerator.device} get outputs end")
            logits = outputs.logits
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), 
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()

            loss = loss / args.gradient_accumulation_steps
            # print(f"epoch {epoch} step {step} device {accelerator.device} calc loss end, loss shape: {loss.shape}, loss = {loss.item()}")
            accelerator.backward(loss)

            if (step > 0 and step % args.gradient_accumulation_steps == 0) or step == len(train_loader) - 1:
                # print(f"epoch {epoch} step {step} device {accelerator.device} step start")
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # print(f"epoch {epoch} step {step} device {accelerator.device} step end")
                completed_steps += 1
                
            if accelerator.is_main_process and step % args.log_interval == 0:
                logger.info(f"epoch {epoch} step {step} / {len(train_loader)} loss: {loss:.5f} lr: {lr_scheduler.get_last_lr()[0]:.7f}") 

            if (completed_steps > 0 and completed_steps % args.save_interval == 0) or completed_steps == 50: # save the first 50 steps just for test
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

            # print(f"epoch {epoch} step {step} device {accelerator.device} end")
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
    print(f"max len meet: {train_ds.max_len_meet}")
    pass




import argparse    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid LLava SFT")
    parser.add_argument("--output_dir", type=str, default="/root/hybridGLAandNSA/ckpts_sft_llava")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hfdataset_path", type=str, default="/root/hybridGLAandNSA/llava-mixed-dataset2")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=3000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()


    main(args)