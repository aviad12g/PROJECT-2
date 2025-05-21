---
**Deterministic Category-Based Sampling (DCBS)** is a *decoding-time algorithm and evaluation harness* for large-language models.  
It clusters the vocabulary, selects the highest-mass category, then greedily takes the top-probability token inside that category—yielding deterministic yet diverse text.  
This repo includes:
* `dcbs.py` – the algorithm + strategy wrapper  
* `run_dcbs_eval.py` – ARC-Easy evaluation with chain-of-thought generation  
* utilities for cluster generation, plotting, docker, and tests  
Until publication, **keep the repository private**.
---

# Deterministic Category-Based Sampling (DCBS)

## What is this project?

This project implements Deterministic Category-Based Sampling (DCBS), a novel text generation technique that provides the semantic diversity benefits of sampling while maintaining deterministic outputs. DCBS works by first identifying the highest-probability token category, then deterministically selecting the best token from within that category. This approach is particularly useful for applications requiring reproducible outputs with controlled diversity.

⚠️ Until publication please keep this repository private.

## Why Deterministic?

Deterministic Category-Based Sampling (DCBS) provides consistent outputs by first selecting the highest-probability token cluster, then deterministically choosing the most likely token within that cluster. This greedy-then-greedy approach ensures reproducible results while maintaining semantic diversity.

## Architecture

The evaluation pipeline consists of three main components:

1. **prepare_arc.py**: Loads ARC-Easy dev examples, wraps each prompt with "Let's think step by step", and outputs a JSONL file for inference.
2. **run_dcbs_eval.py**: Runs inference using Deterministic Category-Based Sampling (DCBS) with the meta-llama/Llama-3.2-1B-Instruct model and records accuracy and latency.
3. **plot_results.py**: Generates plots showing the relationship between accuracy and latency.

## Chat Template Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.dcbs import DeterministicCategoryStrategy, generate_with_strategy

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Apply chat template
messages = [
    {"role": "system", "content": "You are an expert tutor."},
    {"role": "user", "content": "What is photosynthesis?"},
    {"role": "assistant", "content": "Let's think step by step."}
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

# Generate with DCBS
strategy = DeterministicCategoryStrategy(cluster_ids)
output_ids = generate_with_strategy(model, tokenizer, input_ids, strategy, max_new_tokens=100)
```

## Development

### Installation

```bash
pip install -r requirements.txt
```

### Generate Cluster IDs

Before running the evaluation, you need to generate cluster IDs for your model:

```bash
python generate_clusters.py --model meta-llama/Llama-3.2-1B-Instruct
```

### Running Tests

The tests verify the correct behavior of the Deterministic Category-Based Sampling implementation and the chain-of-thought generation:

```bash
python -m unittest discover tests
```

### Running the Evaluation

To prepare ARC-Easy dataset:

```bash
python prepare_arc.py
```

To run evaluation with DCBS:

```bash
python run_dcbs_eval.py --config configs/study_config.yaml
```

To check caching performance:

```bash
python run_dcbs_eval.py --config configs/study_config.yaml --check-cache
```

To generate plots:

```bash
python plot_results.py --results_dir results
```

## Prerequisites

1. Google Cloud project with billing enabled
2. Google Cloud CLI (`gcloud`) installed and configured
3. Docker installed locally

## Setup and Deployment

### 1. Clone and Navigate to Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Configure GCS Bucket

Make sure your Google Cloud Storage bucket exists. The default bucket name is `arc-easy-bucket`. You can specify a different bucket name when running the deployment script.

To create a bucket:
```bash
gsutil mb -l us-central1 gs://arc-easy-bucket
```

### 3. Run Deployment Script

Make the deploy script executable:
```bash
chmod +x deploy.sh
```

Run the deployment script with default settings:
```bash
./deploy.sh
```

Or with a custom bucket name:
```bash
./deploy.sh --bucket=my-custom-bucket
```

The script will:
- Enable required Google Cloud services
- Create a Docker repository if it doesn't exist
- Create the GCS bucket if it doesn't exist
- Build and push the Docker container
- Submit a Vertex AI custom job with a T4 GPU

## Monitoring and Results

### 1. Check Job Status

Monitor the status of your Vertex AI job:
```bash
gcloud ai custom-jobs list --region=us-central1
```

Get detailed information about a specific job:
```bash
gcloud ai custom-jobs describe JOB_ID --region=us-central1
```

### 2. Retrieve Results

After job completion, check the generated files:
```bash
gsutil ls gs://arc-easy-bucket/dcbs-results/
```

Download all results:
```bash
gsutil cp -r gs://arc-easy-bucket/dcbs-results .
```

Or download specific files:
```bash
gsutil cp gs://arc-easy-bucket/dcbs-results/exp_sweeps.csv .
gsutil cp -r gs://arc-easy-bucket/dcbs-results/figures .
```

### 3. View Results

The results include:
- `exp_sweeps.csv`: CSV file with evaluation metrics
- `figures/`: Directory containing visualization plots:
  - `dcbs_accuracy.png`: DCBS accuracy results
  - `dcbs_latency.png`: DCBS latency analysis

## Customization

To customize the evaluation:

1. **Model**: Edit `configs/study_config.yaml` to change the model, decoding parameters, or prompt template.
2. **Prompting**: Modify the reasoning prefix in `configs/study_config.yaml` or the prompt construction in `prepare_arc.py`.
3. **Hardware**: Change the machine type and accelerator in `deploy.sh` for different performance characteristics. 