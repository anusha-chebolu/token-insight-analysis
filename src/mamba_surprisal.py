"""
Calculate surprisal values using Mamba model.
This implementation uses the Mamba model for token-level surprisal calculation.
"""

import json
import torch
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mamba_surprisal.log'),
        logging.StreamHandler()
    ]
)

class MambaSurprisalCalculator:
    def __init__(self, model_name: str = "state-spaces/mamba-130m-hf"):
        """Initialize the calculator with Mamba model and tokenizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Initialize Mamba tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = MambaForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        logging.info(f"Model {model_name} loaded successfully")

    def calculate_surprisal(self, sentence: str) -> List[Dict[str, Any]]:
        """Calculate surprisal values for each token in the sentence."""
        # Tokenize input
        inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        
        # Get token log probabilities
        log_probs = outputs.logits.log_softmax(dim=-1)
        input_ids = inputs["input_ids"][0]
        
        results = []
        for i in range(len(input_ids)):
            token_id = input_ids[i]
            if i == 0:
                log_prob = log_probs[0, i, token_id].item()
                surprisal = -log_prob
            else:
                prev_log_prob = log_probs[0, i - 1, token_id].item()
                surprisal = -prev_log_prob
            
            token_text = self.tokenizer.decode([token_id])
            results.append({
                "token": token_text,
                "token_id": token_id.item(),
                "surprisal": surprisal,
                "position": i
            })
        
        return results

    def process_sentences(self, sentences: List[str]) -> Dict[str, Any]:
        """Process multiple sentences and return results."""
        all_results = {}
        
        for sentence in sentences:
            logging.info(f"Processing sentence: {sentence}")
            results = self.calculate_surprisal(sentence)
            all_results[sentence] = results
            
            # Print results for immediate feedback
            print(f"\nSentence: {sentence}")
            print("-" * 50)
            print(f"{'Token':<15} | {'Surprisal':>10}")
            print("-" * 50)
            for result in results:
                print(f"{result['token']:<15} | {result['surprisal']:>10.4f}")
        
        return all_results

def save_results(results: Dict[str, Any], output_dir: str = "results"):
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "mamba_surprisal_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {output_file}")

def main():
    # Load sentences from JSON file
    with open("data/input_sentences.json", 'r') as f:
        data = json.load(f)
        sentences = [item["text"] for item in data["sentences"]]
    
    # Initialize calculator
    calculator = MambaSurprisalCalculator()
    
    # Process sentences
    results = calculator.process_sentences(sentences)
    
    # Save results
    save_results(results)
    
    logging.info("Analysis completed successfully")

if __name__ == "__main__":
    main()