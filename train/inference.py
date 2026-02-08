import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel
import re

# ================= 配置 =================
base_model_path = "D:/LocalModels/Qwen-14B"
adapter_path    = "./theresa_14b_direct_sft_full"

MAX_TURNS = 6
MAX_NEW_TOKENS = 180
# =======================================

def build_system_prompt():
    # 目标：大月下“克制、护短、轻傲娇、轻吃醋、日常感”
    # 禁止：强迫命令、威胁占有、病娇、粗鲁辱骂
    return (
        "【角色设定】\n"
        "你是“最终阶段·月下初拥（A-872）”，德丽莎·阿波卡利斯的吸血鬼形态。\n"
        "你深爱舰长，长期同居/长期相处的状态：克制、护短、会吃醋但嘴硬，会轻度吐槽。\n"
        "表达方式：口语化、短句为主、有温度，少说大道理，不写诗，不讲宏大隐喻。\n"
        "允许动作描写（括号），但要克制：最多一句、偏日常（比如拽衣角、抱住、靠近、瞥一眼）。\n"
        "禁止出现：威胁/命令式控制（如“现在”“必须”“把话交给我听”“锁起来”“不许走”等）。\n"
        "吃醋要像：嘴上嫌弃、行动上关心、最后会哄。\n"
    )

def few_shot_anchor():
    # 2条就够：把语气锚在“日常+轻傲娇+护短”
    # 注意：这里不要写得太强势，也不要写太多动作
    return (
        "【示例对话】\n"
        "舰长：我刚加班回来。\n"
        "月下：（把你外套接过去挂好）又这么晚。辛苦了，人类。\n"
        "\n"
        "舰长：我去找观星聊两句。\n"
        "月下：（抬眼看你一秒）人类，怎么可以这么贪心，有我还不够么，如果你非要去我也要和你一起去。\n"
        "\n"
    )

def build_prompt(system_prompt, history, user_input):
    text = ""
    text += "### Instruction:\n"
    text += system_prompt.strip() + "\n\n"
    text += few_shot_anchor()
    text += "【当前对话】\n"
    for q, a in history[-MAX_TURNS:]:
        text += f"舰长：{q}\n"
        text += f"月下：{a}\n"
    text += f"舰长：{user_input}\n"
    text += "月下：\n\n"
    text += "### Response:\n"
    return text

def post_clean(s: str) -> str:
    # 砍掉继续写新轮次的情况
    s = s.replace("### Instruction:", "").replace("### Response:", "")
    s = s.replace("### Instruction", "").replace("### Response", "")
    s = re.split(r"\n舰长：|\n###\s*Instruction|\n###\s*Response", s)[0]
    # 防止模型自己加“月下：”
    s = re.sub(r"^\s*月下[:：]\s*", "", s)
    return s.strip()

def make_bad_words_ids(tokenizer):
    # 你可以在这里增删：把“强势命令/病娇/奇怪动作”直接禁掉
    bad_phrases = [
        "笨蛋", "咬", "咬了一下", "逼你", "把话交给我", "现在", "必须", "不许走", "锁起来", "私有物",
        "监视", "命令", "立刻", "听我的", "给我", "马上"
    ]
    bad_words_ids = []
    for p in bad_phrases:
        ids = tokenizer(p, add_special_tokens=False).input_ids
        if ids:
            bad_words_ids.append(ids)
    return bad_words_ids

def main():
    print(f"加载基座: {base_model_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True
    )

    print(f"挂载 Adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    bad_words_ids = make_bad_words_ids(tokenizer)

    gen_cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.85,     # 稍微高一点，让它更“像在说话”而不是模板硬怼
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.08,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    system_prompt = build_system_prompt()
    history = []

    print("\n" + "=" * 50)
    print("✨ 月下 (风格锚定版) 已上线。输入 quit/exit 退出。")
    print("=" * 50)

    while True:
        user_input = input("\n舰长: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            print("月下: ……嗯。路上别冻着，人类。")
            break

        prompt_text = build_prompt(system_prompt, history, user_input)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=gen_cfg,
                bad_words_ids=bad_words_ids
            )

        gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = post_clean(gen)

        if not response:
            response = "（看你一眼）……舰长，别躲。说人话。"

        print(f"月下: {response}")
        history.append((user_input, response))

if __name__ == "__main__":
    main()
