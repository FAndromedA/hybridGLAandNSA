# hybridGLAandNSA
/root/.cache/huggingface/accelerate/default_config.yaml

train first stage with: accelerate launch --multi_gpu --num_processes=3 --gpu_ids=0,1,5 training/train_hybrid.py

nohup accelerate launch --multi_gpu --num_processes=3 --gpu_ids=0,1,5 training/train_hybrid.py &

finally train first stage(distill and pretrain) only on 1 3090 GPU with 32 hours :

CUDA_VISIBLE_DEVICES=7 setsid nohup python training/train_hybrid.py &

train the second stage(sft) with:
setsid nohup accelerate launch --gpu_ids=3,0,1,2 training/train_sft.py yamls/train_sft.yaml &

the nohup.out log start from line: 139795

train the third stage(dpo) with: (do not use zero2 with offload, because: https://github.com/deepspeedai/DeepSpeed/issues/5241)
(actually, we use stage 0 of deepspeed) 

TORCH_DISTRIBUTED_DEBUG=DETAIL setsid nohup accelerate launch --gpu_ids=2,3,4,5 training/train_dpo.py yamls/train_dpo.yaml &

evaluate language model with :
accelerate launch --gpu_ids=5,4,6,7 lm_eval --model hf --model_args pretrained=/root/hybridGLAandNSA/ckpts_train_dpo/checkpoint-33342 --tasks hellaswag --output_path ./eval_out/hybrid --batch_size auto


pretrain llava with :
setsid nohup accelerate launch --gpu_ids=1,0,2,3 training/llava_pretrain.py &

the nohup.out log start from line: 6990

train llava sft:
TORCH_DISTRIBUTED_DEBUG=DETAIL setsid nohup accelerate launch --gpu_ids=1,0,2,3 training/llava_sft.py &

