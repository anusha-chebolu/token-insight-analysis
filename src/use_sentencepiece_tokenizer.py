"""
Use trained SentencePiece tokenizers to encode text.
Requires: pip install sentencepiece
"""

import sentencepiece as spm
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

class SentencePieceTokenizer:
    def __init__(self, model_path):
        """Initialize the SentencePiece tokenizer with a trained model."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        logging.info(f"Loaded SentencePiece model from {model_path}")

    def encode_text(self, text):
        """Encode text into pieces and ids."""
        pieces = self.sp.EncodeAsPieces(text)
        ids = self.sp.EncodeAsIds(text)
        return {
            "pieces": pieces,
            "ids": ids
        }

def process_sentences(tokenizer, sentences):
    """Process multiple sentences and return results."""
    results = {}
    for sentence in sentences:
        logging.info(f"Processing: {sentence}")
        encoded = tokenizer.encode_text(sentence)
        
        # Print results
        print(f"\nSentence: {sentence}")
        print("-" * 80)
        print("Pieces:", encoded["pieces"])
        print("IDs:", encoded["ids"])
        
        results[sentence] = encoded
    
    return results

def main():
    # Load the trained tokenizer
    model_path = "trained_tokenizers/en_512000.model"  # Adjust path as needed
    tokenizer = SentencePieceTokenizer(model_path)
    
    # Load test sentences
    with open("data/input_sentences.json", 'r') as f:
        data = json.load(f)
        sentences = [item["text"] for item in data["sentences"]]
    
    # Process sentences
    results = process_sentences(tokenizer, sentences)
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "sentencepiece_tokenization_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info("Processing completed successfully")

if __name__ == "__main__":
    main()