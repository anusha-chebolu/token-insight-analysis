"""
Compare manual vs direct tokenization approaches.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def direct_tokenization(sentence: str):
    """Direct tokenization using GPT-2's tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()
    
    print("\nDirect Tokenization:")
    print("-" * 70)
    print(f"{'Token':<15} | {'Surprisal':>10}")
    print("-" * 70)
    
    # Tokenize entire sentence at once
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    log_probs = outputs.logits.log_softmax(dim=-1)
    
    direct_results = []
    for i, token in enumerate(tokens):
        if i == 0:  # Skip BOS token
            continue
        surprisal = -log_probs[0, i-1, inputs["input_ids"][0, i]].item()
        print(f"{token:<15} | {surprisal:>10.4f}")
        direct_results.append({
            "token": token,
            "surprisal": surprisal
        })
    
    return direct_results

def manual_tokenization(sentence: str):
    """Manual word-by-word tokenization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()
    
    print("\nManual Tokenization:")
    print("-" * 70)
    print(f"{'Word':<15} | {'Tokens':<30} | {'Surprisal':>10}")
    print("-" * 70)
    
    words = sentence.split()
    context = ""
    
    manual_results = []
    for word in words:
        context += word + " "
        inputs = tokenizer(context, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        
        log_probs = outputs.logits.log_softmax(dim=-1)
        last_token_id = inputs["input_ids"][0, -1]
        surprisal = -log_probs[0, -1, last_token_id].item()
        
        word_tokens = tokenizer.tokenize(word)
        print(f"{word:<15} | {' '.join(word_tokens):<30} | {surprisal:>10.4f}")
        
        manual_results.append({
            "word": word,
            "tokens": word_tokens,
            "surprisal": surprisal
        })
    
    return manual_results

def save_results(results: dict, output_dir: str = "results"):
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "manual_direct_token_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {output_file}")

def main():
    sentence = "I love eating pizza"
    print(f"\nAnalyzing sentence: {sentence}")
    
    # Get results from both methods
    direct_results = direct_tokenization(sentence)
    manual_results = manual_tokenization(sentence)
    
    # Combine results
    results = {
        "sentence": sentence,
        "direct_tokenization": direct_results,
        "manual_tokenization": manual_results
    }
    
    # Save results
    save_results(results)
    
    logging.info("Analysis completed successfully")

if __name__ == "__main__":
    main() 