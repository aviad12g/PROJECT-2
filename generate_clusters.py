#!/usr/bin/env python
# Script to generate cluster IDs for Deterministic Category-Based Sampling

import os
import torch
from transformers import AutoTokenizer
import argparse

def generate_clusters(model_name):
    """Generate simple character-based clusters for a model's tokenizer."""
    print(f"Generating cluster IDs for {model_name}")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    # Generate simple clusters based on first character of each token
    cluster_ids = torch.zeros(vocab_size, dtype=torch.long)
    
    for i in range(vocab_size):
        try:
            token = tokenizer.decode([i])
            # Simple clustering by first character
            if token and len(token) > 0:
                cluster_ids[i] = ord(token[0]) % 100
            else:
                cluster_ids[i] = 0
        except:
            # If decoding fails, assign to cluster 0
            cluster_ids[i] = 0
    
    # Save clusters
    output_path = "data/cluster_ids.pt"
    torch.save(cluster_ids, output_path)
    print(f"Saved {len(torch.unique(cluster_ids))} clusters to {output_path}")
    
    return cluster_ids

def main():
    parser = argparse.ArgumentParser(description="Generate cluster IDs for DCBS")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model name or path")
    args = parser.parse_args()
    
    generate_clusters(args.model)

if __name__ == "__main__":
    main() 