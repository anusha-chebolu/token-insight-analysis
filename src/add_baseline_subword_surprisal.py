"""
Add combined surprisal values for subworded tokens from baseline results.
This script reads baseline_surprisal_results.json and combines surprisal
values for tokens that belong to the same word.
"""

import json
from pathlib import Path
import logging
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SubwordSurprisalAggregator:
    def __init__(self):
        """Initialize the aggregator."""
        self.baseline_results = self.load_baseline_results()

    def load_baseline_results(self) -> Dict[str, Any]:
        """Load baseline surprisal results from JSON file."""
        results_path = Path("results/baseline_surprisal_results.json")
        if not results_path.exists():
            raise FileNotFoundError("Baseline results file not found")
        
        with open(results_path, 'r') as f:
            return json.load(f)

    def is_continuation_token(self, token: str) -> bool:
        """Check if token is a continuation (doesn't start with space)."""
        return not token.startswith(' ') and token != '.'

    def combine_subword_surprisals(self, sentence_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine surprisal values for subworded tokens."""
        combined_results = []
        current_word = {
            "tokens": [],
            "token_ids": [],
            "surprisals": [],
            "combined_surprisal": 0.0
        }

        for token_info in sentence_results:
            token = token_info["token"]
            
            # If token is a continuation, add to current word
            if self.is_continuation_token(token):
                current_word["tokens"].append(token)
                current_word["token_ids"].append(token_info["token_id"])
                current_word["surprisals"].append(token_info["surprisal"])
                current_word["combined_surprisal"] += token_info["surprisal"]
            else:
                # If we have a previous word, save it
                if current_word["tokens"]:
                    combined_results.append({
                        "word": "".join(current_word["tokens"]),
                        "subword_tokens": current_word["tokens"],
                        "token_ids": current_word["token_ids"],
                        "subword_surprisals": current_word["surprisals"],
                        "combined_surprisal": current_word["combined_surprisal"]
                    })
                
                # Start new word
                current_word = {
                    "tokens": [token],
                    "token_ids": [token_info["token_id"]],
                    "surprisals": [token_info["surprisal"]],
                    "combined_surprisal": token_info["surprisal"]
                }

        # Add last word if exists
        if current_word["tokens"]:
            combined_results.append({
                "word": "".join(current_word["tokens"]),
                "subword_tokens": current_word["tokens"],
                "token_ids": current_word["token_ids"],
                "subword_surprisals": current_word["surprisals"],
                "combined_surprisal": current_word["combined_surprisal"]
            })

        return combined_results

    def process_all_sentences(self) -> Dict[str, Any]:
        """Process all sentences from baseline results."""
        subword_results = {}
        
        for sentence, results in self.baseline_results.items():
            logging.info(f"Processing sentence: {sentence}")
            combined = self.combine_subword_surprisals(results)
            
            # Print results for immediate feedback
            print(f"\nSentence: {sentence}")
            print("-" * 80)
            print(f"{'Word':<20} | {'Subword Tokens':<30} | {'Subword Surprisals':<20} | {'Combined':>10}")
            print("-" * 80)
            
            for result in combined:
                tokens_str = " + ".join(result["subword_tokens"])
                surprisals_str = " + ".join(f"{s:.2f}" for s in result["subword_surprisals"])
                print(f"{result['word']:<20} | {tokens_str:<30} | {surprisals_str:<20} | {result['combined_surprisal']:>10.4f}")
            
            subword_results[sentence] = combined

        return subword_results

def save_results(results: Dict[str, Any], output_dir: str = "results"):
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "add_baseline_subword_surprisal_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {output_file}")

def main():
    # Initialize aggregator
    aggregator = SubwordSurprisalAggregator()
    
    # Process all sentences
    results = aggregator.process_all_sentences()
    
    # Save results
    save_results(results)
    
    logging.info("Analysis completed successfully")

if __name__ == "__main__":
    main()