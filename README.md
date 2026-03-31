# Diversity-Driven FunSearch Enhancement

## Environment

- Recommended Python version: Python 3.10

## Installation

Install dependencies from each `requirements.txt` file:

```bash
pip install -r code/requirements.txt
pip install -r code_new/requirements.txt
```

## Manual Configuration

Before running experiments, you need to manually configure the following:

1. Weights & Biases entity
- `code/record_wandb.py`
- `code_new/record_wandb.py`

2. API key settings in FunSearch implementation
- `code/funsearch/implementation/code_embedding.py`
- `code/funsearch/implementation/sampler.py`
- `code_new/funsearch/implementation/code_embedding.py`
- `code_new/funsearch/implementation/sampler.py`

## Project Structure

- `code/`: FunSearch benchmark version
- `code_new/`: Our semantic-based method version

## Run

```bash
cd code
python main.py

cd ../code_new
python main.py
```
