import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import os

# Set the Hugging Face cache directory
os.environ['TRANSFORMERS_CACHE'] = '/N/slate/srcheb/huggingface_cache'

# Model ID
model_id = "shenzhi-wang/Llama3-70B-Chinese-Chat"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
).eval()

# Tokenized garden-path sentences
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
    
    for sentence_id, words in enumerate(sentences, start=1):
        sentence_str = ''.join(words)  # Full sentence for reference
        
        for word_id, word in enumerate(words, start=1):
            # Construct input string up to current word
            current_input = ''.join(words[:word_id])
            tokenized_input = tokenizer(current_input, return_tensors="pt", add_special_tokens=False)
            
            # Move tensors to same device as model
            tokenized_input = {k: v.to(model.device) for k, v in tokenized_input.items()}
            
            with torch.no_grad():
                outputs = model(**tokenized_input)
            
            logits = outputs.logits  # Shape: [1, seq_len, vocab_size]
            probs = torch.softmax(logits, dim=-1)

            # Get the last token ID in the input
            last_token_id = tokenized_input["input_ids"][0, -1]
            last_prob = probs[0, -1, last_token_id]
            surprisal = -torch.log(last_prob)

            result.append({
                "Sentence_ID": sentence_id,
                "Sentence": sentence_str,
                "Word_ID": word_id,
                "Word": word,
                "Surprisal value": surprisal.item()
            })

    return result

# Run surprisal computation
output = compute_word_surprisal(Tokenized_Sents, model, tokenizer)

# Save results to CSV
csv_filename = "../results/chinese_llama3_70B_surprisal.csv"
os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=output[0].keys())
    writer.writeheader()
    writer.writerows(output)

print(f"✅ Results saved to: {csv_filename}")
