# Token Insight Analysis

A project for analyzing tokenization patterns and issues in GPT-2 model, focusing on different types of sentences including garden path sentences and complex structures.

## Project Structure

```
token_insight_analysis/
├── data/               # Input data and datasets
├── src/               # Source code
├── results/           # Analysis results and outputs
├── notebooks/         # Jupyter notebooks for analysis
├── venv/              # Virtual environment
└── requirements.txt   # Project dependencies
```

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses a curated dataset of sentences categorized into:
- Simple sentences
- Complex sentences
- Garden path sentences

The dataset is stored in `data/input_sentences.json` with metadata about each sentence.

## Analysis Focus

This project will analyze:
- Tokenization patterns in GPT-2
- Differences in tokenization between sentence types
- Special cases in garden path sentences
- Token boundary analysis
- Surprisal calculations

## Next Steps

1. Implement tokenization analysis
2. Create visualization tools
3. Analyze patterns across different sentence types
4. Generate insights and reports 