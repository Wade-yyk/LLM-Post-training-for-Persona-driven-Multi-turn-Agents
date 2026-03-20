import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset
import os 
import gc

# ================= 配置区域 =================
# 1. 本地基座模型路径
model_path = "Path-to-your-model"

# 2. 数据文件
data_file = "path-to-your-data"              

# 3. 输出路径
output_dir = "path-to-where-you-want-to-put"               

MAX_SEQ_LENGTH = 2048 
# ===========================================

# 清理内存
gc.collect()
torch.cuda.empty_cache()

print(f"CUDA Available: {torch.cuda.is_available()}")

# 1. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

# 2. 加载模型 (4-bit)
print(f"\n--- 正在加载 14B 基座 ---")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config, 
    device_map="auto",
    attn_implementation="sdpa",
    local_files_only=True 
)
base_model = prepare_model_for_kbit_training(base_model)

# 3. LoRA 配置 (针对 14B 优化)
lora_config = LoraConfig(
    r=64, 
    lora_alpha=32, # alpha 设为 r 的一半或相等，增强学习力度
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# 4. 数据处理
print(f"\n--- 加载数据: {data_file} ---")
dataset = load_dataset("json", data_files={"train": data_file}, split="train")

# def format_tokenize(example):
#     # 纯净的 Prompt 格式
#     prompt = f"### Instruction:\n{example['instruction']}\n\n### Response (月下):\n{example['output']}<|endoftext|>"
#     tokenized = tokenizer(
#         prompt,
#         truncation=True,
#         max_length=MAX_SEQ_LENGTH,
#         padding=False, 
#     )
#     tokenized["labels"] = tokenized["input_ids"].copy()
#     return tokenized

def format_tokenize(example):
    inst = example["instruction"]
    out  = example["output"]

    prompt = f"### Instruction:\n{inst}\n\n### Response:\n"
    full = prompt + out + tokenizer.eos_token

    tok_prompt = tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)
    tok_full   = tokenizer(full,   truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)

    input_ids = tok_full["input_ids"]
    labels = input_ids.copy()

    # mask 掉 prompt 部分
    prompt_len = len(tok_prompt["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len

    return {"input_ids": input_ids, "attention_mask": tok_full["attention_mask"], "labels": labels}


tokenized_dataset = dataset.map(format_tokenize, remove_columns=dataset.column_names)

# 5. 训练参数 (激进微调版)
training_args = TrainingArguments(
    output_dir=output_dir,            
    num_train_epochs=4,            
    per_device_train_batch_size=4,   
    gradient_accumulation_steps=4,   
    gradient_checkpointing=True,     
    learning_rate=1e-4,           
    logging_steps=5,
    save_strategy="epoch",
    optim="paged_adamw_32bit",       
    bf16=True,
    tf32=True,
    group_by_length=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,    
    remove_unused_columns=False,
    warmup_ratio=0.03,
    weight_decay=0.0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)
)

print("\n--- 开始纯净 SFT 训练 ---")
print("这次我们只教它怎么对话，不教它写小说。")
trainer.train()

print(f"\n✅ 训练完成！Adapter 已保存至: {output_dir}")
model.save_pretrained(output_dir)

