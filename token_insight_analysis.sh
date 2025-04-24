#!/bin/bash

#SBATCH -J token_insight_analysis
#SBATCH -p gpu
#SBATCH -A r01024
#SBATCH -o token_insight_analysis.txt
#SBATCH -e token_insight_analysis.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srcheb@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --mem=100G

# Load necessary modules
module load python/gpu/3.11.5
module load git

# Activate virtual environment (make sure this venv uses Python 3.11.5)
source /N/slate/srcheb/token-insight-analysis/venv/bin/activate

# Optional: Upgrade pip
pip install --upgrade pip

# Fix 1: Disable flash attention if not needed
export DISABLE_FLASH_ATTN=1

# Fix 2: Install HuggingFace Transformers (with no flash-attn dependencies)
pip install --no-deps git+https://github.com/huggingface/transformers@main

# Fix 3: Install other requirements
pip install -r requirements.txt

# Run your script
python src/mamba_surprisal.py
