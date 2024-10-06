import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from prompt import ms_v1


def get_data_from_dataloader(dataloader, columns_name):
    all_data = []
    for batch in dataloader:
        batch_size = len(batch[list(batch.keys())[0]])
        for i in range(batch_size):
            tmp = {}
            for col in columns_name:
                tmp[col] = batch[col][i]
            all_data.append(tmp)
    return all_data
    
def convert_data(all_data, columns_name):
    convert_data = {feat: [] for feat in columns_name}
    for record in all_data:
        for feat in columns_name:
            convert_data[feat].append(record[feat])
    return convert_data

def parse():
    args = argparse.ArgumentParser()
    
    args.add_argument('--model_name_or_path', type=str)
    args.add_argument('--dataset_name', type=str)
    args.add_argument('--report_to', type=str)
    args.add_argument('--learning_rate', type=float)
    args.add_argument('--per_device_train_batch_size', type=int)
    args.add_argument('--gradient_accumulation_steps', type=int)
    args.add_argument('--output_dir', type=str)
    args.add_argument('--logging_steps', type=int, default=50)
    args.add_argument('--num_train_epochs', type=int)
    args.add_argument('--max_steps', type=int, default=-1)
    args.add_argument('--lr_scheduler_type', type=str)
    args.add_argument('--warmup_steps', type=int)
    args.add_argument('--gradient_checkpointing', action='store_true')
    args.add_argument('--trust_remote_code', action='store_true')
    
    return args.parse_args()
    
def _filter(example):
    if example['language'] == 'chinese':
        return example

# convert to chinese label
def convert(label):
    if label == 0:
        zh_label='积极'
    elif label==1:
        zh_label='中立'
    elif label==2:
        zh_label='消极'
    return zh_label

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['text'])):
        text = ms_v1.format(input=example['text'][i], answer=convert(example['label'][i]))
        output_texts.append(text)
    return output_texts

def process_func(example):
    """
    Credit: https://github.com/datawhalechina/self-llm.git
    """
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(ms_v1.split('{answer}')[0].format(input=example['text']), add_special_tokens=False)
    response = tokenizer(f"{convert(example['label'])}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


if __name__ == '__main__':
    args = parse()

    # initial process group
    dist.init_process_group('nccl', init_method="env://")
    
    # set rank 
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        save_steps=50,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=True,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        report_to=args.report_to
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    ################
    # Dataset
    ################
    datasets = load_dataset(args.dataset_name).filter(_filter)['train']
    columns_name = datasets.features
    
    # initial DistributedSampler and DataLoader
    sampler = DistributedSampler(datasets, shuffle=True, seed=0, drop_last=False)
    loader = DataLoader(datasets, batch_size=args.per_device_train_batch_size, sampler=sampler)
    
    """
    
    I load the dataset using load_dataset,
    but DistributedSampler can only set into DataLoader,
    so when dataload finished split the dataset,
    I transform the type Dataload -> Dataset to align the type what Trainer need
    
    """
    
    # converting data
    all_data = get_data_from_dataloader(loader, columns_name)
    convert_ = convert_data(all_data, columns_name)
    hf_dataset = Dataset.from_dict(convert_)
    tokenized_id = hf_dataset.map(process_func, remove_columns=hf_dataset.column_names)
    
    # model & DDP
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).cuda()
    model = DDP(model, device_ids=[local_rank])
    
    #  solve the problem about "AttributeError: 'DistributedDataParallel' object has no attribute 'gradient_checkpointing_enable'"
    model = model.module
    
    ################
    # Training
    ################
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)
    trainer.train()

    # # Save and push to hub
    trainer.save_model(train_args.output_dir)
        