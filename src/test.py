def compute_word_surprisal(sentences, model, tokenizer):
    """
    Calculate surprisal values for each word in a list of sentences.
    
    Parameters:
    - sentences: list of lists, where each inner list represents a sentence segmented into words.
                 e.g., [["Word1", "Word2"], ["Word1", "Word2"]]
    - model: the language model used for calculating surprisal.
    - tokenizer: tokenizer compatible with the language model.
    
    Returns:
    - result: list of dictionaries with keys "Sentence_ID", "Sentence", "Word_ID", "Word", "Surprisal value".
    """
    result = []
    
    for sentence_id, words in enumerate(sentences, start=1):
        sentence_str = ''.join(words)  # Combine words without extra spaces
        surprisal_list = []
        
        for word_id, word in enumerate(words, start=1):
            current_input = ''.join(words[:word_id])
            
            # Tokenize the current input up to the current word
            input_ids = torch.tensor([[tokenizer(w).input_ids[0] for w in words[:word_id]]])
            
            # Forward pass through the model without gradient computation
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                
            # Extract probabilities and compute surprisal
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            last_token_id = input_ids[0, -1]
            surprisal = -torch.log(probabilities[0, -1, last_token_id])
            
            # Append details to the result list
            result.append({
                "Sentence_ID": sentence_id,
                "Sentence": sentence_str,
                "Word_ID": word_id,
                "Word": word,
                "Surprisal value": surprisal.item()
            })
    
    return result