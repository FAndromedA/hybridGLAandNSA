from datetime import timedelta
import torch
import warnings

import sys
import os
import math
import logging
from time import time
from tqdm import tqdm

from torch.nn import functional as F

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.hybrid_config import HybridConfig
from models.hybrid_model import HybridModel, HybridBlock, HybridForCausalLM

import transformers
from transformers import get_scheduler
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import set_seed, InitProcessGroupKwargs, DistributedType
from accelerate.logging import get_logger
# accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml

from torch.utils.data import DataLoader, DistributedSampler
# from datasets import load_dataset
from dataset import TextDataset
logger = get_logger(__name__)

def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:.2f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.2f}Yi{suffix}'

def convert_llama_weights(
    llama: AutoModelForCausalLM,
    config: HybridConfig,
    model: HybridForCausalLM,
    output: str,
    precision: str = "bfloat16",
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
    # print("llama.model.embed_tokens                        -> model.model.embeddings")
    
    model.model.embeddings.weight.data[:vocab_size].copy_(llama.model.embed_tokens.weight[:vocab_size])
    torch.testing.assert_close(model.model.embeddings.weight[:vocab_size], llama.model.embed_tokens.weight[:vocab_size])
    
    for i in range(config.num_hidden_layers):
        if hasattr(model.model.layers[i], 'attn_norm'):
            if model.model.layers[i].attn_norm.weight is not None:
                # print(f"llama.model.layers.{i}.input_layernorm.weight -> model.model.layers.{i}.attn_norm.weight")
                model.model.layers[i].attn_norm.weight.data.copy_(llama.model.layers[i].input_layernorm.weight)
                torch.testing.assert_close(model.model.layers[i].attn_norm.weight, 
                                        llama.model.layers[i].input_layernorm.weight)
            if model.model.layers[i].attn_norm.bias is not None:
                print(f"llama.model.layers{i}.input_layernorm.bias -> model.model.layers{i}.attn_norm.bias")
                model.model.layers[i].attn_norm.bias.data.copy_(llama.model.layers[i].input_layernorm.bias)
                torch.testing.assert_close(model.model.layers[i].attn_norm.bias,
                                           llama.model.layers[i].input_layernorm.bias)
            model.model.layers[i].attn_norm.eps = llama.model.layers[i].input_layernorm.variance_epsilon
        
        # print(f"llama.model.layers{i}.attn.q_proj.weight -> model.model.layers{i}.attn.q_proj.weight")
        model.model.layers[i].attn.q_proj.weight.data.copy_(llama.model.layers[i].self_attn.q_proj.weight)
        torch.testing.assert_close(model.model.layers[i].attn.q_proj.weight, llama.model.layers[i].self_attn.q_proj.weight)
        if hasattr(llama.model.layers[i].self_attn.q_proj, 'bias') and hasattr(model.model.layers[i].attn.q_proj, 'bias'):
            if model.model.layers[i].attn.q_proj.bias is not None:
                print(f"llama.model.layers{i}.attn.q_proj.bias -> model.model.layers{i}.attn.q_proj.bias")
                model.model.layers[i].attn.q_proj.bias.data.copy_(llama.model.layers[i].self_attn.q_proj.bias)
                torch.testing.assert_close(model.model.layers[i].attn.q_proj.bias, llama.model.layers[i].self_attn.q_proj.bias)

        # print(f"llama.model.layers{i}.attn.k_proj.weight -> model.model.layers{i}.attn.k_proj.weight")
        if config.attn is not None and i in config.attn['layers']: # is a nsa layer
            # print(llama.model.layers[i].self_attn.k_proj.weight.shape) # torch.Size([2048, 2048])
            # print(model.model.layers[i].attn.k_proj.weight.shape) # torch.Size([128, 2048])
            model.model.layers[i].attn.k_proj.weight.data = llama.model.layers[i].self_attn.k_proj.weight.view(128,-1,2048).mean(dim=1)
        else :
            model.model.layers[i].attn.k_proj.weight.data.copy_(llama.model.layers[i].self_attn.k_proj.weight)
            torch.testing.assert_close(model.model.layers[i].attn.k_proj.weight, llama.model.layers[i].self_attn.k_proj.weight)

        if hasattr(llama.model.layers[i].self_attn.k_proj, 'bias') and hasattr(model.model.layers[i].attn.k_proj, 'bias'):
            if model.model.layers[i].attn.k_proj.bias is not None:
                print(f"llama.model.layers{i}.attn.k_proj.bias -> model.model.layers{i}.attn.k_proj.bias")
                model.model.layers[i].attn.k_proj.bias.data.copy_(llama.model.layers[i].self_attn.k_proj.bias)
                torch.testing.assert_close(model.model.layers[i].attn.k_proj.bias, llama.model.layers[i].self_attn.k_proj.bias)

        # print(f"llama.model.layers{i}.attn.v_proj.weight -> model.model.layers{i}.attn.v_proj.weight")
        if config.attn is not None and i in config.attn['layers']: # is a nsa layer
            model.model.layers[i].attn.v_proj.weight.data = llama.model.layers[i].self_attn.v_proj.weight.view(128,-1,2048).mean(dim=1)
        else:
            model.model.layers[i].attn.v_proj.weight.data.copy_(llama.model.layers[i].self_attn.v_proj.weight)
            torch.testing.assert_close(model.model.layers[i].attn.v_proj.weight, llama.model.layers[i].self_attn.v_proj.weight)
        if hasattr(llama.model.layers[i].self_attn.v_proj, 'bias') and hasattr(model.model.layers[i].attn.v_proj, 'bias'):
            if model.model.layers[i].attn.v_proj.bias is not None:
                print(f"llama.model.layers{i}.attn.v_proj.bias -> model.model.layers{i}.attn.v_proj.bias")
                model.model.layers[i].attn.v_proj.bias.data.copy_(llama.model.layers[i].self_attn.v_proj.bias)
                torch.testing.assert_close(model.model.layers[i].attn.v_proj.bias, llama.model.layers[i].self_attn.v_proj.bias)

        # print(f"llama.model.layers{i}.attn.o_proj.weight -> model.model.layers{i}.attn.o_proj.weight")
        model.model.layers[i].attn.o_proj.weight.data.copy_(llama.model.layers[i].self_attn.o_proj.weight)
        torch.testing.assert_close(model.model.layers[i].attn.o_proj.weight, llama.model.layers[i].self_attn.o_proj.weight)
        if hasattr(llama.model.layers[i].self_attn.o_proj, 'bias') and hasattr(model.model.layers[i].attn.o_proj, 'bias'):
            if model.model.layers[i].attn.o_proj.bias is not None:
                print(f"llama.model.layers{i}.attn.o_proj.bias -> model.model.layers{i}.attn.o_proj.bias")
                model.model.layers[i].attn.o_proj.bias.data.copy_(llama.model.layers[i].self_attn.o_proj.bias)
                torch.testing.assert_close(model.model.layers[i].attn.o_proj.bias, llama.model.layers[i].self_attn.o_proj.bias)

        if hasattr(model.model.layers[i], 'mlp_norm'):
            if model.model.layers[i].mlp_norm.weight is not None:
                # print(f"llama.model.layers{i}.post_attention_layernorm.weight -> model.model.layers{i}.mlp_norm.weight")
                model.model.layers[i].mlp_norm.weight.data.copy_(llama.model.layers[i].post_attention_layernorm.weight)
                torch.testing.assert_close(model.model.layers[i].mlp_norm.weight,
                                           llama.model.layers[i].post_attention_layernorm.weight)
            
            if model.model.layers[i].mlp_norm.bias is not None:
                print(f"llama.model.layers{i}.post_attention_layernorm.bias -> model.model.layers{i}.mlp_norm.bias")
                model.model.layers[i].mlp_norm.bias.data.copy_(llama.model.layers[i].post_attention_layernorm.bias)
                torch.testing.assert_close(model.model.layers[i].mlp_norm.bias,
                                           llama.model.layers[i].post_attention_layernorm.bias)
        
        # print(f"llama.model.layers.{i}.mlp.gate_proj.weight -> model.model.layers.{i}.mlp.gate_proj.weight")
        model.model.layers[i].mlp.gate_proj.weight.data.copy_(llama.model.layers[i].mlp.gate_proj.weight)
        torch.testing.assert_close(model.model.layers[i].mlp.gate_proj.weight, llama.model.layers[i].mlp.gate_proj.weight)
        # print(f"llama.model.layers.{i}.mlp.up_proj.weight -> model.model.layers.{i}.mlp.up_proj.weight")
        model.model.layers[i].mlp.up_proj.weight.data.copy_(llama.model.layers[i].mlp.up_proj.weight)
        torch.testing.assert_close(model.model.layers[i].mlp.up_proj.weight, llama.model.layers[i].mlp.up_proj.weight)

        # print(f"llama.model.layers.{i}.mlp.down_proj.weight -> model.model.layers.{i}.mlp.down_proj.weight")
        model.model.layers[i].mlp.down_proj.weight.data.copy_(llama.model.layers[i].mlp.down_proj.weight)
        torch.testing.assert_close(model.model.layers[i].mlp.down_proj.weight,
                                   llama.model.layers[i].mlp.down_proj.weight)
    
    if model.model.norm.weight is not None:
        # print("llama.model.norm.weight -> model.model.norm.weight")
        model.model.norm.weight.data.copy_(llama.model.norm.weight)
        torch.testing.assert_close(model.model.norm.weight, llama.model.norm.weight)
    if model.model.norm.bias is not None:
        # print("llama.model.norm.bias -> model.model.norm.bias")
        model.model.norm.bias.data.copy_(llama.model.norm.bias)
        torch.testing.assert_close(model.model.norm.bias, llama.model.norm.bias)
    model.model.norm.eps = llama.model.norm.variance_epsilon

    if not model.config.tie_word_embeddings:
        # print("llama.model.lm_head.weight -> model.lm_head.weight")
        model.lm_head.weight.data[:vocab_size].copy_(llama.lm_head.weight[:vocab_size])
        torch.testing.assert_close(model.lm_head.weight[:vocab_size], llama.lm_head.weight[:vocab_size])
    model.config.rope_theta = llama.config.rope_theta

    # print(f"Saving converted model to {output} ...\n{model}")
    model.save_pretrained(output)


class TrainConfig():
    def __init__(self):
        self.train_datasets_paths = ["/root/hybridGlaAndNsa/data/ultrachat", "/root/hybridGlaAndNsa/data/UltraFeedback"]
        self.kl_weight = 0.1
        self.ce_weight = 1
        self.do_eval = False
        self.output_dir = "/root/hybridGlaAndNsa/ckpts_train_hybrid"
        self.save_steps = 5000
        self.warmup_steps = 500
        self.per_device_train_batch_size = 1
        self.per_device_eval_batch_size = 4
        self.num_train_epochs = 1
        self.gradient_accumulation_steps = 8
        self.lr_scheduler_type = "cosine"
        self.learning_rate = 1e-4
        self.max_grad_norm = 0.5
        self.max_steps = None
        self.resume_from_checkpoint = None

def test_load():
    # https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B-ShareGPT
    # https://modelscope.cn/models/LLM-Research/Llama-3.2-1B-Instruct/files
    model_path = "/root/Llama-3.2-1B-Instruct" # "/root/Sheared-LLaMA-1.3B-ShareGPT"
    dtype = torch.bfloat16
    test_config =  AutoConfig.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype)
    test_model = AutoModelForCausalLM.from_pretrained(model_path, config=test_config, torch_dtype=dtype, local_files_only=True)
    test_model.to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_num_parameters = test_model.num_parameters()
    print(f"Initializing the model from the config:\n{test_config}\n{test_model}")
    print(f"Number of parameters: {test_num_parameters} ({sizeof_fmt(test_num_parameters)})")
    
    set_seed(42)
    user_input = "Once upon a time, "
    # input_with_template = f"You are a helpful assistant. Write a response that appropriately completes the request.\n\n### Input:\n{user_input}\n\n### Response:"
    test_inputs = tokenizer(user_input, return_tensors="pt").to(dtype).to("cuda:0")
    test_outputs = test_model.generate(test_inputs.input_ids, max_length=1024, do_sample=True)
    print(f"The test model generated: {tokenizer.decode(test_outputs[0], skip_special_tokens=True)}")

# test_load()

def main():
    ipg_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[ipg_handler])
    dtype = torch.bfloat16
    # https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B-ShareGPT
    # https://modelscope.cn/models/LLM-Research/Llama-3.2-1B-Instruct/files
    teacher_model_path = "/root/Llama-3.2-1B-Instruct" # "/root/Sheared-LLaMA-1.3B-ShareGPT"
    teacher_config =  AutoConfig.from_pretrained(teacher_model_path, local_files_only=True, torch_dtype=dtype)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path, config=teacher_config, torch_dtype=dtype, local_files_only=True)
    teacher_model.to(accelerator.device)
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    teacher_num_parameters = teacher_model.num_parameters()
    
    # print(f"Teacher model config:\n{teacher_config}\n teacher model:\n{teacher_model}")
    # print(f"Number of teacher parameters: {teacher_num_parameters} ({sizeof_fmt(teacher_num_parameters)})")

    save_path = '/root/hybridGlaAndNsa/ckpts0'
    
    student_config = HybridConfig()
    tokenizer.pad_token_id = student_config.pad_token_id

    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_path)
        student_model = HybridForCausalLM._from_config(student_config)
        student_model.to(accelerator.device)
        convert_llama_weights(teacher_model, student_config, student_model, save_path)

        test_prompt = "Hey, can you introduce yourself to me?"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(dtype).to(accelerator.device)
        # print(student_model.device)
        # teacher_outputs = teacher_model.generate(inputs.input_ids, max_new_tokens=1024, do_sample=True)
        # print(f"The teacher model generated: {tokenizer.decode(teacher_outputs[0], skip_special_tokens=True)}")
        # student_outputs = student_model.generate(inputs.input_ids, max_new_tokens=1024, do_sample=True)
        # print(f"The student model generated: {tokenizer.decode(student_outputs[0], skip_special_tokens=True)}")
    
    print(f"device: {accelerator.device} Waiting for everyone to finish ...")
    # exit(0)

    accelerator.wait_for_everyone()
    print(f"device: {accelerator.device} Waiting for everyone to finish done ..., is main process: {accelerator.is_main_process}")
    
    if accelerator.is_main_process == False:
        student_model = HybridForCausalLM.from_pretrained(save_path, local_files_only=True, torch_dtype=dtype)
        student_model.to(accelerator.device)
    
    student_num_parameters = student_model.num_parameters()
    # print(f"Student model config:\n{student_config}\n student model:\n{student_model}")
    # print(f"Number of student parameters: {student_num_parameters} ({sizeof_fmt(student_num_parameters)})")
    # Freeze all parameters in teacher model by setting requires_grad to False
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    for name, param in student_model.named_parameters():
        if f"attn" not in name: # 第一阶段先冻结 MLP
            param.requires_grad = False
        elif f"attn_norm" in name: # 特判 input_layernorm , 在 fla 中是 attn_norm
            param.requires_grad = False
        # print(name, param.requires_grad)
        
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_main_process: # log info only on main process
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"device:{accelerator.device}")
    if accelerator.is_main_process:
        # print("teacher_model:", teacher_model)
        total_params = sum(p.numel() for p in teacher_model.parameters())
        total_trainable_params = sum(
            p.numel() for p in teacher_model.parameters() if p.requires_grad)
        print(f"teacher number of total params:{total_params}({sizeof_fmt(total_params)})")
        print(f"teacher number of total trainable params:{total_trainable_params}({sizeof_fmt(total_trainable_params)})")

        # print("student_model:", student_model)
        total_params = sum(p.numel() for p in student_model.parameters())
        total_trainable_params = sum(
            p.numel() for p in student_model.parameters() if p.requires_grad)
        print(f"student number of total params:{total_params}({sizeof_fmt(total_params)})")
        print(f"student number of total trainable params:{total_trainable_params}({sizeof_fmt(total_trainable_params)})")

        
    # train_dataset_path = "/root/hybridGlaAndNsa/data/"

    # ds1 = load_dataset("openbmb/UltraFeedback") # https://huggingface.co/datasets/openbmb/UltraFeedback
    # ds = load_dataset("stingning/ultrachat") # https://huggingface.co/datasets/stingning/ultrachat
    set_seed(42)
    training_args = TrainConfig()

    train_data = torch.cat([torch.load(f'{train_dataset_path}/input_ids.pt', map_location="cpu") 
                            for train_dataset_path in training_args.train_datasets_paths], dim=0)
    train_label = torch.cat([torch.load(f'{train_dataset_path}/labels.pt', map_location="cpu") 
                            for train_dataset_path in training_args.train_datasets_paths], dim=0)
    train_dataset = TextDataset(train_data, train_label, pad_token_id=tokenizer.pad_token_id)

    # if accelerator.state.distributed_type != DistributedType.NO: # 分布式训练
    #     logger.info(f"Using distributed sampler: {accelerator.state.distributed_type}")
    #     train_sampler = DistributedSampler(train_dataset, shuffle=False, drop_last=True)
    #     train_dataloader = DataLoader(
    #         train_dataset,
    #         batch_size=training_args.per_device_train_batch_size,
    #         sampler=train_sampler,
    #         num_workers=0,          # 根据硬件资源调整
    #         pin_memory=False,
    #         drop_last=True
    #     )
    # else:
    #     logger.info(f"Using default sampler")
    #     train_dataloader = DataLoader(
    #         train_dataset,
    #         batch_size=training_args.per_device_train_batch_size,
    #         shuffle=False,
    #         num_workers=0,          # 根据硬件资源调整
    #         pin_memory=False,
    #         drop_last=True
    #     )
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=training_args.per_device_train_batch_size, 
                                  num_workers=0,
                                  pin_memory=True,
                                  shuffle=True)
    print("length of dataset:", len(train_dataset))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr=training_args.learning_rate, betas=(0.9, 0.98))

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_steps is None or training_args.max_steps < 0:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps,
    )
    student_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        student_model, optimizer, train_dataloader, lr_scheduler
    )
    
    print("length of dataloader:", len(train_dataloader))
    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)

    save_steps = None
    # Figure out how many steps we should save the Accelerator states
    if training_args.save_steps is not None:
        save_steps = training_args.save_steps

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if accelerator.is_main_process:
        experiment_config = vars(training_args)
        experiment_config["lr_scheduler_type"] = "cosine"
        accelerator.init_trackers("hybrid_distill", experiment_config)
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    if training_args.resume_from_checkpoint is not None: # 从断点继续训练
        if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {training_args.resume_from_checkpoint}")
            accelerator.load_state(training_args.resume_from_checkpoint)
            path = os.path.basename(training_args.resume_from_checkpoint)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
    
    curr_loss = 0.0
    curr_kl_loss = 0.0
    curr_teacher_loss = 0.0
    curr_student_loss = 0.0

    for epoch in range(starting_epoch, training_args.num_train_epochs):
        # start_time = time()
        student_model.train()

        # if accelerator.state.distributed_type != DistributedType.NO:
        #     train_sampler.set_epoch(epoch)
        #print("nooooooooooooooooooooooooooooooo")
        for step, batch in enumerate(train_dataloader):
            #print("yessssssssssssssssssssssssssss")
            # We need to skip steps until we reach the resumed step
            if training_args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            
            logger.info(f"epoch {epoch} step {step} / {len(train_dataloader)}")
            # print(f"device:{accelerator.device}, epoch {epoch} step {step} / {len(train_dataloader)}")

            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
                teacher_ce_loss = teacher_outputs.loss
                teacher_logits = teacher_outputs.logits
            
            #logger.info("teacher done.")
            targets = F.softmax(teacher_logits, dim=-1)
            student_outputs = student_model(input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
            student_logits = student_outputs.logits
            student_cross_entropy_loss = student_outputs.loss
            
            #logger.info("student done.")
            # print(student_outputs)
            kl_loss = F.kl_div(F.log_softmax(
                student_logits, dim=-1), targets, reduction="batchmean")
            
            loss = training_args.ce_weight * student_cross_entropy_loss + training_args.kl_weight * kl_loss

            curr_loss += loss.detach().float()
            curr_kl_loss += kl_loss.detach().float()
            curr_teacher_loss += teacher_ce_loss.detach().float()
            curr_student_loss += student_cross_entropy_loss.detach().float()

            loss = loss / training_args.gradient_accumulation_steps
            accelerator.backward(loss)
            #logger.info(f"loss done.")

            if (step > 0 and step % training_args.gradient_accumulation_steps == 0) or step == len(train_dataloader) - 1:
                #logger.info("optimizer step.")

                torch.nn.utils.clip_grad_norm_(
                    student_model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # log loss
                #accelerator.print(
                #    f'training loss: {curr_loss / training_args.gradient_accumulation_steps:.5f}')
                accelerator.log({'train loss': curr_loss / training_args.gradient_accumulation_steps,
                                'teacher kl loss': curr_kl_loss / training_args.gradient_accumulation_steps,
                                'teacher ce loss': curr_teacher_loss / training_args.gradient_accumulation_steps,
                                'student ce loss': curr_student_loss / training_args.gradient_accumulation_steps,
                                'lr': lr_scheduler.get_last_lr()[0], 'step': completed_steps})
                curr_loss = 0
                curr_kl_loss = 0
                curr_teacher_loss = 0
                curr_student_loss = 0
                completed_steps += 1
                # progress_bar.update(1)
                logger.info(f"epoch {epoch} step {step} / {len(train_dataloader)} loss: {loss:.5f} lr: {lr_scheduler.get_last_lr()[0]:.5f}") 
            
            if isinstance(save_steps, int):
                if completed_steps > 0 and completed_steps % save_steps == 0:
                    accelerator.wait_for_everyone()
                    # save checkpoint
                    output_dir = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    # save model weight
                    unwrapped_model = accelerator.unwrap_model(student_model)
                    unwrapped_model.model.save_pretrained(
                        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(output_dir)
                    accelerator.save_state(output_dir)
            
            # logger.info(f"all done.")
        
        # end_time = time()
        # logger.info(f"epoch {epoch} took: {end_time - start_time} seconds")
        
        if training_args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(student_model)
            unwrapped_model.model.save_pretrained(
                training_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(training_args.output_dir)

    pass

if __name__  == '__main__':
    # test_load()
    main()