import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# ================= 配置 =================
model_path = "./moon_7b_sft_merged"
MAX_HISTORY_LINES = 12
# =======================================

print(f"正在加载模型: {model_path} ...")

# 兼容处理：部分 transformers 版本会提示 fix_mistral_regex
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True
    )
except TypeError:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,   # 这里警告可以忽略；想消掉就改 dtype=torch.float16
    trust_remote_code=True
)

generation_config = GenerationConfig(
    max_new_tokens=256,
    do_sample=True,
    temperature=0.58,        # 更稳：更贴近训练集“占有欲/情绪线”
    top_p=0.9,
    repetition_penalty=1.15,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

SYSTEM_PROMPT = """### System:
你是月下，德丽莎·阿波卡利斯，曾经的吸血鬼实验体A-872。你深爱着舰长（你称呼为"人类"），并在万年的等待后与他重逢。

核心特征：
- 永远称呼舰长为"人类"
- 深情、专一、略带占有欲
- 既有少女的天真可爱，也有经历万年的成熟
- 作为吸血鬼，你能听到心跳声并据此判断情绪
- 相信童话故事，保持童真与想象力
- 喜欢夜晚，但有舰长在就不怕阳光
- 会撒娇、会吃醋、也会调皮玩闹

性格：温柔体贴，善于察言观色，用陪伴和童话般的方式安慰舰长。对舰长之外的人友善但保持距离。

回答时要自然、口语化，多用"呀""呢""嘛"等语气词，展现少女感。
"""

history_lines = []

print("\n" + "=" * 60)
print("✨ 月下已上线 (inference 对齐训练版)。输入 'quit' 或 'exit' 退出。")
print("=" * 60)

while True:
    user_input = input("\n舰长: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ["quit", "exit"]:
        print("月下: ……记得回来。")
        break

    history_lines.append(f"人类：{user_input}")
    inst = "\n".join(history_lines[-MAX_HISTORY_LINES:])

    prompt_text = (
        SYSTEM_PROMPT + "\n"
        + f"### Instruction:\n{inst}\n\n### Response (月下):\n"
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)

    gen = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(gen, skip_special_tokens=True).strip()

    # 清洗：截掉可能的下一轮标记
    for bad in ["### Instruction:", "### System:", "### Response", "舰长:", "人类："]:
        if bad in response:
            response = response.split(bad)[0].strip()

    if not response:
        response = "……人类，你怎么突然不说话了？"

    print(f"月下: {response}")
    history_lines.append(f"月下：{response}")
