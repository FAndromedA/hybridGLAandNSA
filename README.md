# hybridGLAandNSA


train first stage with: accelerate launch --multi_gpu --num_processes=3 --gpu_ids=0,1,5 training/train_hybrid.py

nohup accelerate launch --multi_gpu --num_processes=3 --gpu_ids=0,1,5 training/train_hybrid.py &

CUDA_VISIBLE_DEVICES=7 setsid nohup python training/train_hybrid.py &

```
accelerate config                                                                                            
--------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?                                                                                                     
This machine                                                                                                                                      
--------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?                                                                                                              
multi-GPU                                                                                                                                         
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1                                                        
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: no                 
Do you wish to optimize your script with torch dynamo?[yes/NO]:no                                                                                 
Do you want to use DeepSpeed? [yes/NO]: yes                                                                                                       
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: no                                                                            
--------------------------------------------------------------------------------------------------------------------------------------------------What should be your DeepSpeed's ZeRO optimization stage?                                                                                          
2                                                                                                                                                 
--------------------------------------------------------------------------------------------------------------------------------------------------Where to offload optimizer states?                                                                                                                
cpu                                                                                                                                               
--------------------------------------------------------------------------------------------------------------------------------------------------Where to offload parameters?                                                                                                                      
none                                                                                                                                              
How many gradient accumulation steps you're passing in your script? [1]: 8                                                                        
Do you want to use gradient clipping? [yes/NO]: yes                                                                                               
What is the gradient clipping value? [1.0]: 1.0                                                                                                   
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: no
Do you want to enable Mixture-of-Experts training (MoE)? [yes/NO]: no
How many GPU(s) should be used for distributed training? [1]:3
--------------------------------------------------------------------------------------------------------------------------------------------------Do you wish to use mixed precision?
bf16                                                                                                                                              
accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml
```