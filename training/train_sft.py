import os
import sys
import random
import logging

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoConfig, set_seed
from tqdm import tqdm

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from trl import SFTTrainer, setup_chat_format

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.hybrid_config import HybridConfig
from models.hybrid_model import HybridModel, HybridForCausalLM

import torch.distributed as dist
from datetime import timedelta

logger = logging.getLogger(__name__)

from typing import Any, List, Dict, Tuple, Optional, NewType
from dataclasses import dataclass, field

@dataclass
class SFTDistillConfig(SFTConfig):
    """
    Arguments related to the distillation process.
    """

    with_distill: bool = field(
        default=True,
        metadata={"help": "Whether we have the first stage of distillation."},
    )
    prev_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the previous distilled model in this progressive distillation."},
    )
    nsa_layers: List[int] = field(
        default=None,
        metadata={"help": "List of NSA layers."},
    )
    init_with_kqvo: bool = field(default=False, metadata={"help": "Whether to init with transformer weights."})
    decontaminate: bool = field(default=False, metadata={"help": "Whether to apply the decontaminate steps."})

    teacher_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Teacher Model."})
    # teacher_model_init_kwargs: Optional[Dict[str, Any]] = field(default=None, metadata={"help": "Teacher Model Init Kwargs."})
    teacher_load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the Teacher Model in 8bit."})
    kl_weight: float = field(
        default=0.1,
        metadata={"help": "Ratio of KL loss."},
    )
    ce_weight: float = field(
        default=1,
        metadata={"help": "Ratio of CE loss."},
    )

def main():
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=25000)) # tokenizer takes more than 4 hours
    
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTDistillConfig), description="Fine-tune a model on the H4 dataset.")
    model_args, data_args, training_args = parser.parse()
    
    training_args.packing = True
    # print(model_args, data_args, training_args)

    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")
    
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
    )

    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    tokenizer = get_tokenizer(model_args, data_args)

    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False,
        # use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=HybridConfig.from_pretrained(model_args.model_name_or_path),
        **model_kwargs,
    )
    print(f"Model device: {model.device}")

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Applying chat template",
    )

    ##########################
    # Decontaminate benchmarks, 标准化空格，将不该大写的地方小写，根据一定的 filter 筛除包含某些文字的数据
    ##########################
    if training_args.decontaminate:
        num_raw_train_samples = len(raw_datasets["train"])
        raw_datasets = raw_datasets.filter(decontaminate_humaneval, batched=True, batch_size=10_000, num_proc=data_args.preprocessing_num_workers)
        num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
        logger.info(
            f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
        )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    # 随机采样几个训练集样本输出
    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # dataset_text_field="text", # removed by trl, you should set it in SFTConfig, and it's default value is "text"
        # max_seq_length=training_args.max_seq_length,
        processing_class=tokenizer,
        # packing=True,
        peft_config=get_peft_config(model_args),
        # dataset_kwargs=training_args.dataset_kwargs,
    )
    
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
