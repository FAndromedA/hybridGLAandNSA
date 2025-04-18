import json
import os
import sys
import random
import logging

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_quantization_config,
    get_tokenizer,
    decontaminate_humaneval
)

from transformers import TrainerCallback
from functools import partial # for hooker
from trl import DPOTrainer

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from my_models.hybrid_config import HybridConfig
from my_models.hybrid_model import HybridBlock, HybridModel, HybridForCausalLM

logger = logging.getLogger(__name__)

class NanDebugCallback(TrainerCallback):
    def __init__(self, log_path="nan_debug.log"):
        self.log_path = log_path
        self.record = {}
    
    def _get_tensor_stats(self, tensor, name):
        """获取张量关键统计信息（支持BF16/FP32）"""
        if tensor is None:
            return f"{name}: None"
        
        # 转换到FP32以确保统计计算稳定
        tensor = tensor.detach().cpu().float() if tensor.dtype == torch.bfloat16 else tensor.detach().cpu()
        
        stats = {
            "name": name,
            "shape": list(tensor.shape),
            "has_nan": torch.isnan(tensor).any().item(),
            "has_inf": torch.isinf(tensor).any().item(),
            "max": tensor.max().item() if tensor.numel() > 0 else None,
            "min": tensor.min().item() if tensor.numel() > 0 else None,
            "mean": tensor.mean().item() if tensor.numel() > 0 else None,
            "std": tensor.std().item() if tensor.numel() > 0 else None,
        }
        
        # 对Embedding层添加首行样本
        if "embedding" in name.lower() and tensor.ndim >= 2:
            stats["sample"] = tensor[0].tolist()[:3]  # 首行前3个元素
        return stats

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return

        # 仅检查Embedding层是否包含NaN
        embeddings = model.get_input_embeddings().weight
        if not torch.isnan(embeddings).any():
            return

        # 开始收集全模型参数和梯度信息
        current_step = state.global_step
        print(f"\n⚠️ Step {current_step}: Embedding层检测到NaN！开始收集调试信息...")
        
        debug_info = {
            "step": current_step,
            "embeddings": self._get_tensor_stats(embeddings, "input_embeddings.weight"),
            "parameters": [],
            "gradients": []
        }

        # 遍历所有参数和梯度
        for name, param in model.model.named_parameters():
            debug_info["parameters"].append(
                self._get_tensor_stats(param, f"param.{name}")
            )
            debug_info["gradients"].append(
                self._get_tensor_stats(param.grad, f"grad.{name}")
            )

        # 保存到文件
        with open(self.log_path, "w") as f:
            json.dump(debug_info, f, indent=2)
        
        print(f"✅ 调试信息已保存至 {self.log_path}")
        control.should_training_stop = True  # 建议停止训练以排查问题

class SimpleModelInspector(TrainerCallback):
    def __init__(self, log_path="debug_info.json"):
        self.log_path = log_path
        self.debug_data = []
    
    def _get_tensor_stats(self, tensor, name):
        """获取张量的关键统计信息（兼容BF16）"""
        if tensor is None:
            return {"name": name, "status": "None"}
        
        # 转换到CPU和FP32以确保统计计算安全
        t = tensor.detach().cpu().float() if tensor.dtype == torch.bfloat16 else tensor.detach().cpu()
        
        stats = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "has_nan": torch.isnan(tensor).any().item(),
            "has_inf": torch.isinf(tensor).any().item(),
            "max": t.max().item() if t.numel() > 0 else None,
            "min": t.min().item() if t.numel() > 0 else None,
            "mean": t.mean().item() if t.numel() > 0 else None,
            "std": t.std().item() if t.numel() > 0 else None,
        }
        
        # 对Embedding层增加采样数据
        if "embedding" in name.lower() and tensor.ndim >= 2:
            stats["sample"] = t[:2, :3].tolist()  # 取前两行的前三列
        return stats

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return

        # 仅当检测到Embedding层NaN时触发全量检查
        embeddings = model.get_input_embeddings().weight
        if not torch.isnan(embeddings).any():
            return

        # 开始收集全模型信息
        current_step = state.global_step
        print(f"Step {current_step}: 检测到Embedding层NaN，开始收集全模型状态...")
        
        model_state = {
            "step": current_step,
            "embeddings": self._get_tensor_stats(embeddings, "input_embeddings.weight"),
            "parameters": [],
            "gradients": [],
            "activations": self._get_activations(model)  # 获取最近一次前向的激活值
        }

        # 收集所有参数和梯度
        for name, param in model.model.named_parameters():
            model_state["parameters"].append(self._get_tensor_stats(param, f"param.{name}"))
            model_state["gradients"].append(self._get_tensor_stats(param.grad, f"grad.{name}"))

        self.debug_data.append(model_state)
        
        # 立即写入文件
        with open(self.log_path, "w") as f:
            json.dump(self.debug_data, f, indent=2)
        
        print(f"已保存完整调试信息到 {self.log_path}")
        control.should_training_stop = True  # 可选：是否停止训练

    def _get_activations(self, model):
        """获取最近一次前向传播的激活值（需配合下面的一处模型修改）"""
        if not hasattr(model, "last_activations"):
            return []
        return [
            self._get_tensor_stats(act, name)
            for name, act in model.last_activations.items()
        ]

# 在模型定义中添加以下代码以捕获最后一次前向的激活值
class ActivationCaptureModel(HybridForCausalLM):  # 替换为你的模型类
    def forward(self, *args, **kwargs):
        self.last_activations = {}
        
        def save_activation(name):
            def hook(module, input, output):
                self.last_activations[name] = output.detach()
            return hook
        
        # 仅监控关键层
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Embedding, 
                                 torch.nn.Linear,
                                 torch.nn.LayerNorm)):
                hooks.append(module.register_forward_hook(save_activation(name)))
        
        try:
            output = super().forward(*args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
        
        return output

def main():
    # there are some bugs that the hybrid model return nan
    # ipg_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    # accelerator = Accelerator(kwargs_handlers=[ipg_handler])
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()
    
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

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

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
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
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
    config = HybridConfig.from_pretrained(model_args.model_name_or_path)
    config.use_cache = False
    model = HybridForCausalLM.from_pretrained( # ActivationCaptureModel
        model_args.model_name_or_path,
        config=config,
        **model_kwargs,
    )
    # def nan_hook(name, module, input, output):
        
    #     if isinstance(input, tuple):
    #         for t in input:
    #             if isinstance(t, torch.Tensor) and torch.isnan(t).any():
    #                 print(f"!input NaN in {module.__class__.__name__}, input: {input}, output: {output}")
    #                 raise ValueError(f"input NaN in {module.__class__.__name__}")
    #     elif isinstance(input, torch.Tensor) and torch.isnan(input).any():
    #         print(f"!input NaN in {module.__class__.__name__}, input: {input}, output: {output}")
    #         raise ValueError(f"input NaN in {name}. {module.__class__.__name__}")
        

    #     if isinstance(output, tuple):
    #         for t in output:
    #             if isinstance(t, torch.Tensor) and torch.isnan(t).any():
    #                 print(f"!output NaN in {module.__class__.__name__}, input: {input}, output: {output}")
    #                 raise ValueError(f"output NaN in {module.__class__.__name__}")
    #     elif isinstance(output, torch.Tensor) and torch.isnan(output).any():
    #         print(f"!output NaN in {module.__class__.__name__}, input: {input}, output: {output}")
    #         raise ValueError(f"output NaN in {name}. {module.__class__.__name__}")
    #     # print(f"no NaN in {name}. {module.__class__.__name__}")

    # for name, module in model.model.named_modules():
    #     # print(f"Registering hook for {name}")
    #     hook = partial(nan_hook, name)
    #     module.register_forward_hook(hook)
    
    # model.lm_head.register_forward_hook(partial(nan_hook, 'lm_head'))
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        **model_kwargs,
    )
    ref_model.eval().requires_grad_(False) 

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "dpo",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    num_raw_train_samples = len(raw_datasets["train"])
    raw_datasets = raw_datasets.filter(
        decontaminate_humaneval,
        fn_kwargs={"text_column": "text_chosen"},
        batched=True,
        batch_size=10000,
        num_proc=1,
        desc="Decontaminating HumanEval samples",
    )
    # 实际上是 DPOTrainer 默认是 keep_end，如果有 padding 的话就全部都是 0，导致训练出错。
    # def has_nonzero_attention(example, tokenizer):
    #     prompt_input = tokenizer(
    #         example["text_prompt"],
    #         return_attention_mask=True,
    #         add_special_tokens=False
    #     )
    #     chosen_input = tokenizer(
    #         example["text_chosen"], 
    #         return_attention_mask=True, 
    #         add_special_tokens=False
    #     )
    #     rejected_input = tokenizer(
    #         example["text_rejected"], 
    #         return_attention_mask=True, 
    #         add_special_tokens=False
    #     )
    #     if sum(prompt_input["attention_mask"]) == 0 or sum(chosen_input["attention_mask"]) == 0 or sum(rejected_input["attention_mask"]) == 0:
    #         logger.warning(f"Attention mask all zeros for example: {example}")
    #         return False
    #     return (
    #         sum(prompt_input["attention_mask"]) > 0 and
    #         sum(chosen_input["attention_mask"]) > 0 and
    #         sum(rejected_input["attention_mask"]) > 0
    #     )

    # raw_datasets = raw_datasets.filter(
    #     has_nonzero_attention,
    #     fn_kwargs={"tokenizer": tokenizer},
    #     desc="Filtering samples with attention_mask all zeros",
    #     num_proc=data_args.preprocessing_num_workers,
    # )
    
    num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
    logger.info(
        f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.6f}%) samples from the training set."
    )
    
    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]
    # # 随机采样几个训练集样本输出
    # for index in random.sample(range(len(raw_datasets["train"])), 3):
    #     logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
    #     logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
    #     logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    # 检查tokenized后的样本
    # sample = tokenizer(train_dataset[0]["prompt"], return_tensors="pt").to('cuda')
    # logger.info(f"Tokenized sample input_ids: {sample['input_ids']}, length: {sample['input_ids'].shape}")
    # logger.info(f"Tokenized sample attention_mask: {sample['attention_mask']}")

    # # 检查模型输出
    # with torch.no_grad():
    #     ref_model.to('cuda')
    #     print(f"Ref Model device: {ref_model.device}")
    #     outputs = ref_model(**sample)
    #     logger.info(f"Model logits sample: {outputs.logits}")
    model.train()

    trainer = DPOTrainer(
        model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        # peft_config=get_peft_config(model_args),
    )
    # trainer.add_callback(SimpleModelInspector(log_path="nan_debug.json"))

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

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Training complete! ***")

if __name__ == '__main__':
    main()