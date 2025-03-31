"""
Calculate surprisal values using trained unigram tokenizer with mamba model.
"""

import torch
from tokenizers import Tokenizer
import json
from pathlib import Path
import logging
from transformers import AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)

def load_unigram_tokenizer(tokenizer_path):
    """Load the trained unigram tokenizer."""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer

def load_mamba_model():
    """Load and initialize the mamba model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-2.8b")
    model.to(device)
    model.eval()
    return model

def calculate_surprisal_unigram_mamba(sentence, unigram_tokenizer, mamba_model):
    """Calculate surprisal values using unigram tokenizer and mamba model."""
    # Encode with unigram tokenizer
    encoded = unigram_tokenizer.encode(sentence)
    token_ids = torch.tensor([encoded.ids]).to(mamba_model.device)
    tokens = encoded.tokens

    # Get model outputs
    with torch.no_grad():
        outputs = mamba_model(token_ids, labels=token_ids)

    # Get token log probabilities
    log_probs = outputs.logits.log_softmax(dim=-1)

    print(f"\nSentence: {sentence}")
    print("-" * 50)
    print(f"{'Token':<15} | {'Surprisal':>10}")
    print("-" * 50)

    results = []
    for i in range(len(encoded.ids)):
        token_id = encoded.ids[i]
        if i == 0:
            log_prob = log_probs[0, i, token_id].item()
            surprisal = -log_prob
        else:
            prev_log_prob = log_probs[0, i - 1, token_id].item()
            surprisal = -prev_log_prob

        token_text = tokens[i]
        print(f"{token_text:<15} | {surprisal:>10.4f}")
        
        results.append({
            "token": token_text,
            "token_id": token_id,
            "surprisal": surprisal,
            "position": i
        })

    return results

def main():
    # Load the trained unigram tokenizer
    tokenizer_path = "models/word_unigram_tokenizer.json"
    unigram_tokenizer = load_unigram_tokenizer(tokenizer_path)

    # Load and initialize the mamba model
    mamba_model = load_mamba_model()
    logging.info("Mamba model loaded successfully")

    # Load test sentences
    with open("data/input_sentences.json", 'r') as f:
        data = json.load(f)
        sentences = [item["text"] for item in data["sentences"]]

    # Process each sentence
    all_results = {}
    for sentence in sentences:
        results = calculate_surprisal_unigram_mamba(sentence, unigram_tokenizer, mamba_model)
        all_results[sentence] = results

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "unigram_mamba_surprisal_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    logging.info("Analysis completed successfully")

if __name__ == "__main__":
    main()