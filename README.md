# Distributed-Data-Parallel

# DDP WORKFLOW
1. initial process group
2. torch.cuda.set_device
3. initial Distributedsamlper
4. initial DDP Model


# MODEL & DATASET
1. Model: gpt-2(124M)
2. Dataset: multilingual-sentiments from Huggingface, which use language "chinese"


# CAVEAT
1. model = DDP(model, device_ids=[local_rank])   # local_rank must be integer 
2. training_args.gradient_checkpointing_kwargs={'use_reentrant':False}   # if gradient_checkpointing is True，then you have to add "gradient_checkpointing_kwargs" to training_args
3. DistributedSampler can only set in DataLoader, so you may transform your input data_set to SFTTrainer or Trainer
4. model = model.module  # when you initialed your ddp model, remenber add this command below, or it will raise type of "gradient_checkpointing_enable" problem
