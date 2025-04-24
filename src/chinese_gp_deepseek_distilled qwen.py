import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Set the Hugging Face cache directory
os.environ['TRANSFORMERS_CACHE'] = '/N/slate/srcheb/huggingface_cache'

# DeepSeek Qwen 1.5B distill model
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).eval()
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Tokenized sentences (word-level)
Tokenized_Sents = [
    ['批评', '学生', '的', '教官', '之后', ',', '那位', '督导', '打算', '取消', '这次', '活动'],
    ['举报', '主管', '的', '秘书', '之前', ',', '那个', '员工', '反复', '确认了', '录音', '证据'],
    ['陪伴', '考生', '的', '家长', '之外', ',', '那位', '主任', '还会', '监督', '考场', '秩序'],
    ['出卖', '雇主', '的', '副手', '之后', ',', '那位', '助理', '立即', '公开了', '所有', '资料'],
    ['虐待', '孩子', '的', '保姆', '之后', ',', '那对', '夫妻', '最终', '受到了', '法律', '制裁'],
    ['接送', '旅客', '的', '导游', '之前', ',', '那个', '司机', '给车', '进行了', '彻底', '清洁'],
    ['看望', '病人', '的', '医生', '之前', ',', '那位', '院长', '特意', '慰问了', '退休', '人员'],
    ['袭击', '局长', '的', '秘书', '之后', ',', '那名', '男子', '匆忙', '跑进了', '一家', '餐厅'],
    ['得罪', '领导', '的', '助手', '之后', ',', '那位', '工人', '还未', '察觉', '有何', '不妥'],
    ['面试', '学徒', '的', '师傅', '之前', ',', '那位', '考官', '详细', '说明了', '考察', '内容'],
    ['服侍', '首相', '的', '厨师', '之前', ',', '那个', '青年', '还曾', '当过', '酒店', '前台'],
    ['拜访', '教授', '的', '学生', '之前', ',', '那个', '助教', '精心', '准备了', '一份', '礼物'],
    ['训练', '士兵', '的', '将军', '之后', ',', '那位', '司令', '开车', '回到了', '指挥', '办公室']
]

def compute_word_surprisal(sentences, model, tokenizer):
    result = []
    terminal_log = []

    for sentence_id, words in enumerate(sentences, start=1):
        sentence_str = ''.join(words)
        terminal_log.append(f"Processing Sentence {sentence_id}: {sentence_str}")

        for word_id, word in enumerate(words, start=1):
            if word_id > 8:
                continue

            current_input = ''.join(words[:word_id])
            tokenized_input = tokenizer(current_input, return_tensors="pt", add_special_tokens=False)
            tokenized_input = {k: v.to(model.device) for k, v in tokenized_input.items()}

            with torch.no_grad():
                outputs = model(**tokenized_input)

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            word_tokens = tokenizer(word, add_special_tokens=False)["input_ids"]
            terminal_log.append(f"Word_ID {word_id}: '{word}' | Token IDs: {word_tokens}")

            token_probs = []
            for i in range(-len(word_tokens), 0):
                token_id = tokenized_input["input_ids"][0, i]
                prob = probs[0, i, token_id]
                token_probs.append(prob.item())
                terminal_log.append(f"  Token ID {token_id}: Prob = {prob.item():.6f}")

            word_prob = torch.tensor(token_probs).log().sum().exp()
            surprisal = -torch.log(word_prob)
            surprisal_value = surprisal.item()

            terminal_log.append(f"  Final Surprisal: {surprisal_value:.4f}")

            result.append({
                "Sentence_ID": sentence_id,
                "Sentence": sentence_str,
                "Word_ID": word_id,
                "Word": word,
                "Surprisal value": surprisal_value
            })

    return result, terminal_log

# Run analysis
output, logs = compute_word_surprisal(Tokenized_Sents, model, tokenizer)

# Save CSV
csv_filename = "../results/deepseek_qwen1.5b_surprisal.csv"
os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=output[0].keys())
    writer.writeheader()
    writer.writerows(output)

# Save logs as JSON
log_filename = "../results/deepseek_qwen1.5b_log.json"
with open(log_filename, mode="w", encoding="utf-8") as log_file:
    json.dump(logs, log_file, indent=2, ensure_ascii=False)

print(f"Results saved to: {csv_filename}")
print(f"Terminal log saved to: {log_filename}")


# Load the CSV results
df = pd.read_csv(csv_filename)

# Filter for Word_IDs 1 to 8
df_filtered = df[df["Word_ID"].isin(range(1, 9))]

# Group by Word_ID and calculate average surprisal
avg_surprisal = df_filtered.groupby("Word_ID")["Surprisal value"].mean().reset_index()

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(avg_surprisal["Word_ID"], avg_surprisal["Surprisal value"], width=0.6, color="#4B3C8E")
plt.xlabel("Word Position in Sentence (Word_ID)")
plt.ylabel("Average Surprisal")
plt.title("Average Surprisal for Word Positions 1 to 8")
plt.xticks(avg_surprisal["Word_ID"])
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
plot_path = "../results/deepseek_qwen1.5b_surprisal_barplot.png"
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

print(f"Bar graph saved to: {plot_path}")
