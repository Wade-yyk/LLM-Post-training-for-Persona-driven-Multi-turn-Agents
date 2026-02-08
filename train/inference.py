import re
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig
)
from peft import PeftModel

# =============== 配置 ===============
BASE_MODEL_PATH = "D:/LocalModels/Qwen-14B"
ADAPTER_PATH    = "./theresa_14b_direct_sft_full"

MAX_LINES = 12          # 保留最近多少行（人类/月下都算一行），建议 8~16
MAX_NEW_TOKENS = 220
# ====================================

def post_clean(s: str) -> str:
    """清理模型可能续写的下一轮标签，但别误杀正常内容"""
    s = s.strip()

    # 截掉模型可能开始写下一轮/吐模板
    for bad in ["### Instruction:", "### System:", "### Response", "User:", "Assistant:", "舰长:"]:
        if bad in s:
            s = s.split(bad)[0].strip()

    # 去掉开头可能的“月下：”
    s = re.sub(r"^\s*(月下|Assistant)[:：]\s*", "", s)
    return s.strip()

def load_model_and_tokenizer():
    print(f"加载基座: {BASE_MODEL_PATH}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=True
    )

    print(f"挂载 Adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    return model, tokenizer

# “纯净但有效”的短锚点（不想要就设为空字符串，但会更飘）
SHORT_ANCHOR = "你是月下。你必须称呼对方为“人类”。用第一人称对话。"

def build_prompt(history_lines, user_input: str) -> str:
    """
    history_lines: ["人类：xxx", "月下：yyy", ...]
    对齐训练分布：用 人类：/月下： 行，且用单个 Instruction 包住
    """
    # 先把当前输入写入历史（人类：）
    history_lines.append(f"人类：{user_input}")

    # 截取最近 MAX_LINES 行
    ctx = history_lines[-MAX_LINES:]

    # 拼 Instruction
    inst = "\n".join([SHORT_ANCHOR] + ctx).strip()

    # 对齐训练模板
    return f"### Instruction:\n{inst}\n\n### Response (月下):\n"

def main():
    model, tokenizer = load_model_and_tokenizer()

    gen_cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.6,     # 更稳：逻辑更连贯
        top_p=0.88,
        top_k=50,
        repetition_penalty=1.10,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    history_lines = []

    print("\n" + "=" * 60)
    print("✨ 纯净推理 (短锚点 + 对齐训练模板) 已上线。输入 quit/exit 退出。")
    print("=" * 60)

    while True:
        user_input = input("\n舰长: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            print("Bye.")
            break

        prompt_text = build_prompt(history_lines, user_input)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_cfg)

        gen = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        response = post_clean(gen) or "……"

        print(f"月下: {response}")

        # 把回复写回历史（月下：）
        history_lines.append(f"月下：{response}")

if __name__ == "__main__":
    main()
