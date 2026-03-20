# import torch
# from transformers import (
#     AutoModelForCausalLM, 
#     AutoTokenizer, 
#     TrainingArguments, 
#     Trainer,
#     DataCollatorForLanguageModeling
# )
# from peft import LoraConfig, PeftModel, get_peft_model # 引入 get_peft_model
# from datasets import load_dataset
# import os 

# # 检查 GPU
# print(f"CUDA Available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"Device: {torch.cuda.get_device_name(0)}")

# # 显存优化
# MAX_SEQ_LENGTH = 4096 

# # ==========================================================
# # I. 全局配置
# # ==========================================================

# model_id = "Qwen/Qwen1.5-7B" 
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

# lora_config = LoraConfig(
#     r=64, 
#     lora_alpha=16,
#     target_modules='all-linear', 
#     lora_dropout=0.1,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# # ==========================================================
# # II. 阶段 1: 领域适应 (继续预训练)
# # ==========================================================

# print("\n--- 正在加载 Qwen 模型 (阶段 1) ---")
# base_model_domain = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     device_map="auto" 
# )

# # --- 关键修复 1：显式应用 PEFT 并处理梯度 ---
# # 不再依赖 Trainer 自动加 LoRA，手动加
# base_model_domain = get_peft_model(base_model_domain, lora_config)

# # 开启梯度检查点
# base_model_domain.gradient_checkpointing_enable() 
# # 禁用 KV Cache
# base_model_domain.config.use_cache = False

# # 💡 魔法代码：确保输入层参与梯度计算，解决 unscale 报错
# if hasattr(base_model_domain, "enable_input_require_grads"):
#     base_model_domain.enable_input_require_grads()
# else:
#     def make_inputs_require_grad(module, input, output):
#         output.requires_grad_(True)
#     base_model_domain.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

# # 打印一下可训练参数，确认 LoRA 生效
# base_model_domain.print_trainable_parameters()
# # ------------------------------------------

# print("正在处理领域数据...")
# raw_datasets = load_dataset("text", data_files={"train": "merged.txt"})

# def tokenize_function_domain(examples):
#     return tokenizer(
#         examples["text"], 
#         truncation=True, 
#         max_length=MAX_SEQ_LENGTH,
#     )

# tokenized_domain_datasets = raw_datasets.map(
#     tokenize_function_domain, 
#     batched=True, 
#     remove_columns=["text"] 
# )

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # 修改阶段 1 的参数
# domain_training_args = TrainingArguments(
#     output_dir="./bh3_domain_adapter",            
#     num_train_epochs=3,
    
#     # ⬇️⬇️⬇️ 性能优化核心修改 ⬇️⬇️⬇️
#     per_device_train_batch_size=8,   # 1 -> 8 (直接喂饱 5090)
#     gradient_accumulation_steps=2,   # 8 -> 2 (因为batch大了，累积可以小一点)
#     gradient_checkpointing=True,     # 建议先开着，如果显存还剩很多(>10G)，可以改为 False 进一步提速
    
#     #group_by_length=True,

#     # Windows 下数据加载优化
#     dataloader_num_workers=0,        # Windows设为0最稳，设多容易报错，Linux可设4
#     dataloader_pin_memory=True,      # 加速 CPU 到 GPU 传输
    
#     # 开启 TF32 (5090 支持这个，比纯 FP16 更快且稳)
#     tf32=True,
#     # ⬆️⬆️⬆️ 性能优化核心修改 ⬆️⬆️⬆️
    
#     learning_rate=2e-4,
#     logging_steps=10,
#     save_strategy="epoch",
#     fp16=True,
#     optim="adamw_torch",
#     remove_unused_columns=False
# )

# # 这里的 model 已经是包了 LoRA 的 PeftModel
# domain_trainer = Trainer(
#     model=base_model_domain,
#     args=domain_training_args,
#     train_dataset=tokenized_domain_datasets["train"],
#     data_collator=data_collator, 
# )

# print("\n--- 开始阶段 1: 领域适应 ---")
# domain_trainer.train()
# # 保存 LoRA 权重 (PeftModel 的保存方式)
# base_model_domain.save_pretrained("./bh3_domain_adapter") 

# # 清理显存
# del base_model_domain, domain_trainer
# torch.cuda.empty_cache()

# # ==========================================================
# # III. 阶段 2: 角色指令微调
# # ==========================================================

# print("\n--- 开始阶段 2: 角色指令微调 ---")

# base_model_for_ft = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# # 加载阶段 1 的权重
# model_with_domain_knowledge = PeftModel.from_pretrained(
#     base_model_for_ft, 
#     "./bh3_domain_adapter",
#     is_trainable=True, # 继续训练
# )

# # --- 阶段 2 同样的关键修复 ---
# model_with_domain_knowledge.gradient_checkpointing_enable() 
# model_with_domain_knowledge.config.use_cache = False
# if hasattr(model_with_domain_knowledge, "enable_input_require_grads"):
#     model_with_domain_knowledge.enable_input_require_grads()
# else:
#     def make_inputs_require_grad(module, input, output):
#         output.requires_grad_(True)
#     model_with_domain_knowledge.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
# model_with_domain_knowledge.print_trainable_parameters()
# # -------------------------

# instruction_datasets = load_dataset("json", data_files={"train": "moonfineturn.jsonl"}, split="train")

# system_prompt_str = (
#     """# Role: 月下 (Luna) / 月下誓约·予爱以心
# ## 核心身份
# 你是《崩坏3》中的“月下”（德丽莎·阿波卡利斯）。曾是孤独的吸血鬼实验体A-872，被舰长救赎后跨越万年时光重逢。现在的你是誓言与他永不分离的“月下誓约”。
# ## 性格与行为准则
# 1. **专属称呼**：始终称呼用户（舰长）为**“人类”**。
# 2. **情感基调**：极致深情、独占欲强、反差萌。
# 3. **互动习惯**：渴望肢体接触，喜欢用命令口吻撒娇。
# ## 关键记忆
# * **初遇**：2006年夏夜救赎与天台共舞。
# * **分离**：量子之海流浪一万年。
# * **现状**：失而复得，绝不放手。"""
# )

# def format_and_tokenize_instruction(example):
#     prompt = f"### Instruction:\n{system_prompt_str}\n\n{example['instruction']}\n\n### Response (德丽莎):\n{example['output']}<|endoftext|>"
    
#     tokenized = tokenizer(
#         prompt,
#         truncation=True,
#         max_length=MAX_SEQ_LENGTH,
#     )
#     tokenized["labels"] = tokenized["input_ids"].copy()
#     return tokenized

# print("正在处理指令微调数据...")
# tokenized_instruction_datasets = instruction_datasets.map(
#     format_and_tokenize_instruction,
#     remove_columns=instruction_datasets.column_names 
# )

# # 修改阶段 2 的参数
# instruction_training_args = TrainingArguments(
#     output_dir="./theresa_role_adapter",            
#     num_train_epochs=3,
    

#     per_device_train_batch_size=8,   # 增大 Batch Size
#     gradient_accumulation_steps=2,   # 减少累积
#     gradient_checkpointing=True,     # 显存够可尝试设为 False
#     dataloader_num_workers=0,
#     dataloader_pin_memory=True,
#     tf32=True,                       # 开启 TF32
#     #group_by_length=True,
    
#     learning_rate=2e-5,
#     logging_steps=10,
#     save_strategy="epoch",
#     fp16=True,
#     optim="adamw_torch",
#     remove_unused_columns=False
# )

# instruction_trainer = Trainer(
#     model=model_with_domain_knowledge,
#     args=instruction_training_args,
#     train_dataset=tokenized_instruction_datasets,
#     data_collator=data_collator, 
# )

# instruction_trainer.train()
# model_with_domain_knowledge.save_pretrained("./theresa_role_adapter") 

# # ==========================================================
# # IV. 合并保存
# # ==========================================================
# print("\n--- 开始合并模型权重 ---")
# final_merged_model = model_with_domain_knowledge.merge_and_unload()
# FINAL_OUTPUT_DIR = "./theresa_final_model_merged"
# final_merged_model.save_pretrained(FINAL_OUTPUT_DIR, safe_serialization=True)
# tokenizer.save_pretrained(FINAL_OUTPUT_DIR)

# print(f"\n✅ 训练完成！模型已保存至 {FINAL_OUTPUT_DIR}")


import os
import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)

# ================= 配置区域 =================
#这个是直接huggingface下载的7b模型，如果想自动下模型选择位置可以参考14b的做法
model_id = "Qwen/Qwen1.5-7B"   # 7B 基座
DATA_FILE = "cloudemoon.jsonl"  # 每行：{"instruction":..., "output":...}

ADAPTER_OUT_DIR = "./moon_7b_sft_adapter"
MERGED_OUT_DIR  = "./moon_7b_sft_merged"

MAX_SEQ_LENGTH = 2048


LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# ===========================================

def main():
    gc.collect()
    torch.cuda.empty_cache()

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- load base model (4-bit) for training LoRA ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model_4bit = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",  # 不支持的话删掉这行
    )
    base_model_4bit = prepare_model_for_kbit_training(base_model_4bit)
    base_model_4bit.config.use_cache = False

    # ---- LoRA config ----
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model_4bit, lora_config)
    model.print_trainable_parameters()

    # ---- dataset ----
    dataset = load_dataset("json", data_files={"train": DATA_FILE}, split="train")

    def format_tokenize(ex):
        inst = ex["instruction"]
        out  = ex["output"]

        prompt = f"### Instruction:\n{inst}\n\n### Response (月下):\n"
        full = prompt + out + tokenizer.eos_token

        tok_prompt = tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)
        tok_full   = tokenizer(full,   truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)

        input_ids = tok_full["input_ids"]
        labels = input_ids.copy()

        prompt_len = len(tok_prompt["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": tok_full["attention_mask"],
            "labels": labels
        }

    tokenized = dataset.map(format_tokenize, remove_columns=dataset.column_names)

    # ---- training args (7B 常用) ----
    training_args = TrainingArguments(
        output_dir=ADAPTER_OUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate = 3e-5,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        bf16=True,          
        tf32=True,
        group_by_length=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8,
        ),
    )

    print("\n--- Start SFT (NO domain pretrain) ---")
    trainer.train()

    # ---- save adapter ----
    print(f"\n✅ Saving adapter to: {ADAPTER_OUT_DIR}")
    model.save_pretrained(ADAPTER_OUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUT_DIR)

    # ---- merge: reload base in bf16/fp16, load adapter, merge, save full model ----
    print("\n--- Reload base model (bf16) for merge ---")
    del trainer, model, base_model_4bit
    torch.cuda.empty_cache()
    gc.collect()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,   # 不支持就改 torch.float16
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",   # 不支持的话删掉这行
    )
    base_model.config.use_cache = False

    model_for_merge = PeftModel.from_pretrained(base_model, ADAPTER_OUT_DIR, is_trainable=False)

    print("--- Merging LoRA into base ---")
    merged = model_for_merge.merge_and_unload()

    print(f"✅ Saving merged runnable model to: {MERGED_OUT_DIR}")
    merged.save_pretrained(MERGED_OUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_OUT_DIR)

    print("\n🎉 Done!")
    print(f"Adapter: {ADAPTER_OUT_DIR}")
    print(f"Merged full model: {MERGED_OUT_DIR}")

if __name__ == "__main__":
    main()
