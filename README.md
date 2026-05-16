
# Qwen2.5-1.5B Data Science Benchmarking & Fine-tuning

## Project Overview

This project benchmarks and fine-tunes the Qwen2.5-1.5B language model for data science code generation using the DS-1000 dataset. The workflow includes:

- **Few-shot Prompting:** Created tailored few-shot prompts for each data science framework in DS-1000 (pandas, numpy, matplotlib, sklearn, scipy, pytorch, tensorflow).
- **Sample Generation:** Used LLMs to generate new coding samples for each framework, following the DS-1000 style.
- **Validation:** Automatically validated generated samples for correctness and executability, with framework-specific logic and robust error handling.
- **Training:** Fine-tuned Qwen2.5-1.5B on the validated samples for 3 and 5 epochs, comparing to the base model.
- **Benchmarking:** Evaluated base and fine-tuned models on DS-1000, reporting scores by library and perturbation type.

## Directory Structure

- `generate_samples.py` — Generates new data science coding samples using LLMs and few-shot prompts.
- `validate5.py` — Validates generated samples for correctness and executability.
- `train.py` — Fine-tunes Qwen2.5-1.5B on validated samples.
- `results/` — Contains benchmarking results for base and fine-tuned models.
- `validated/` — Stores validated samples for each framework.
- `data/` — Contains DS-1000 and answer files.
- `complete_checkpoint*/`, `final_model*/` — Model checkpoints and final weights.

## Results

### Overall Scores

| Model                | Mean Score |
|----------------------|------------|
| Qwen2.5-1.5B (base)  | 0.102      |
| Qwen2.5-1.5B 3 epoch | 0.247      |
| Qwen2.5-1.5B 5 epoch | 0.264      |

### By Library (5 Epochs)

| Library     | Score  |
|-------------|--------|
| Matplotlib  | 0.445  |
| Numpy       | 0.305  |
| Pandas      | 0.186  |
| Pytorch     | 0.250  |
| Scipy       | 0.236  |
| Sklearn     | 0.165  |
| Tensorflow  | 0.289  |

### By Perturbation Type (5 Epochs)

| Type               | Score  |
|--------------------|--------|
| Difficult-Rewrite  | 0.117  |
| Origin             | 0.354  |
| Semantic           | 0.248  |
| Surface            | 0.178  |

## How to Use

1. **Generate Samples:**
	- Run `generate_samples.py` for each framework to create new coding problems and solutions.
2. **Validate Samples:**
	- Use `validate5.py` to filter out invalid or non-executable samples.
3. **Fine-tune Model:**
	- Train Qwen2.5-1.5B using `train.py` with the validated data.
4. **Evaluate:**
	- Benchmark the model using the scripts and compare results in `results/`.

## Money Spent

<img width="1556" height="930" alt="image" src="https://github.com/user-attachments/assets/4df86026-3e6a-4e79-8436-fb7716d53183" />

