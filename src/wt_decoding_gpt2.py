"""
Implement word-level tokenization (WT) decoding for GPT-2 model.
This script calculates surprisal values using word-level tokenization approach.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wt_decoding.log'),
        logging.StreamHandler()
    ]
)

class WTDecoder:
    def __init__(self, model_name: str = "gpt2"):
        """Initialize the decoder with GPT-2 model and tokenizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Get space and subword indices
        self.space_idx, self.subword_idx = self._get_space_subword_idx()
        
        logging.info(f"Model {model_name} loaded successfully")

    def _get_space_subword_idx(self) -> Tuple[List[int], List[int]]:
        """Get indices for space and subword tokens."""
        space_idx = []
        subword_idx = []
        
        for token, idx in self.tokenizer.vocab.items():
            if token.startswith("Ġ"):
                space_idx.append(idx)
            else:
                subword_idx.append(idx)
                
        return space_idx, subword_idx

    def _prepare_batch(self, text: str) -> List[Tuple]:
        """Prepare text for model input in batches."""
        batches = []
        tokenizer_output = self.tokenizer(text)
        ids = tokenizer_output.input_ids
        attn = tokenizer_output.attention_mask
        
        # Add BOS token if needed
        if "gpt" in self.model.config.model_type:
            ids = [self.model.config.bos_token_id] + ids
            attn = [1] + attn
            
        start_idx = 0
        ctx_size = self.model.config.max_position_embeddings
        
        # Process in sliding windows
        while len(ids) > ctx_size:
            batches.append((
                {"input_ids": torch.tensor(ids[:ctx_size]).unsqueeze(0).to(self.device),
                 "attention_mask": torch.tensor(attn[:ctx_size]).unsqueeze(0).to(self.device)},
                torch.tensor(ids[1:ctx_size+1]).to(self.device),
                start_idx,
                True
            ))
            
            ids = ids[int(ctx_size/2):]
            attn = attn[int(ctx_size/2):]
            start_idx = int(ctx_size/2)
            
        # Process remaining tokens
        batches.append((
            {"input_ids": torch.tensor(ids).unsqueeze(0).to(self.device),
             "attention_mask": torch.tensor(attn).unsqueeze(0).to(self.device)},
            torch.tensor(ids[1:]).to(self.device),
            start_idx,
            False
        ))
            
        return batches

    def calculate_wt_surprisal(self, sentence: str) -> List[Dict[str, Any]]:
        """Calculate word-level tokenization surprisal values."""
        results = []
        batches = self._prepare_batch(sentence)
        is_continued = False
        
        for batch in batches:
            batch_input, output_ids, start_idx, will_continue = batch
            
            with torch.no_grad():
                model_output = self.model(**batch_input)
                
            toks = self.tokenizer.convert_ids_to_tokens(output_ids)
            index = torch.arange(0, output_ids.shape[0])
            probs = torch.nn.functional.softmax(model_output.logits.squeeze(0), dim=-1)
            all_surp = -1 * torch.log2(probs)
            actual_surp = all_surp[index, output_ids]
            
            for i in range(start_idx, len(toks)):
                cleaned_tok = self.tokenizer.convert_tokens_to_string([toks[i]]).replace(" ", "")
                boundary_type = "B" if toks[i].startswith("Ġ") else "I"
                boundary_prob = torch.log2(torch.sum(probs[i][self.space_idx])).item()
                inside_prob = torch.log2(torch.sum(probs[i][self.subword_idx])).item()
                
                results.append({
                    "token": cleaned_tok,
                    "token_id": output_ids[i].item(),
                    "surprisal": actual_surp[i].item(),
                    "position": i,
                    "boundary_type": boundary_type,
                    "boundary_prob": boundary_prob,
                    "inside_prob": inside_prob
                })
                
            if not is_continued:
                results.append({
                    "token": "<eos>",
                    "token_id": self.model.config.eos_token_id,
                    "surprisal": -1 * torch.log2(torch.sum(probs[-1][self.space_idx])).item(),
                    "position": len(toks),
                    "boundary_type": "B",
                    "boundary_prob": torch.log2(torch.sum(probs[-1][self.space_idx])).item(),
                    "inside_prob": torch.log2(torch.sum(probs[-1][self.subword_idx])).item()
                })
                
            is_continued = will_continue
            
        return results

    def process_sentences(self, sentences: List[str]) -> Dict[str, Any]:
        """Process multiple sentences and return results."""
        all_results = {}
        
        for sentence in sentences:
            logging.info(f"Processing sentence: {sentence}")
            results = self.calculate_wt_surprisal(sentence)
            all_results[sentence] = results
            
            # Print results for immediate feedback
            print(f"\nSentence: {sentence}")
            print("-" * 70)
            print(f"{'Token':<15} | {'Surprisal':>10} | {'Boundary':>8} | {'B-Prob':>10} | {'I-Prob':>10}")
            print("-" * 70)
            for result in results:
                print(f"{result['token']:<15} | {result['surprisal']:>10.4f} | {result['boundary_type']:>8} | {result['boundary_prob']:>10.4f} | {result['inside_prob']:>10.4f}")
        
        return all_results

def save_results(results: Dict[str, Any], output_dir: str = "results"):
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "wt_decoding_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {output_file}")

def main():
    # Load sentences from JSON file
    with open("data/input_sentences.json", 'r') as f:
        data = json.load(f)
        sentences = [item["text"] for item in data["sentences"]]
    
    # Initialize decoder
    decoder = WTDecoder()
    
    # Process sentences
    results = decoder.process_sentences(sentences)
    
    # Save results
    save_results(results)
    
    logging.info("Analysis completed successfully")

if __name__ == "__main__":
    main() 