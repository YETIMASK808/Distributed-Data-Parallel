# Distributed-Data-Parallel

# DDP流程

1. 初始化进程组
2. torch.cuda.set_device
3. 初始化采样器
4. 初始化ddp模型

# 注意事项
1. model = DDP(model, device_ids=[local_rank])  # 里面的local_rank要整数类型
2. training_args.gradient_checkpointing_kwargs={'use_reentrant':False}  #如果gradient_checkpointing为True，则必须新增gradient_checkpointing_kwargs到args里
3. SFTTrainer或者Trainer输入的数据格式都不一样，要注意数据转换
4. model = model.module  # ddp完模型后要加这行命令，不然保存不了checkpoint