#!/usr/bin/env python
# Script to run ARC-Easy evaluation with Deterministic Category-Based Sampling

import argparse
import json
import os
import time
import yaml
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the DCBS implementation
from src.dcbs import DeterministicCategoryStrategy, generate_with_strategy

def process_example(example, model, tokenizer, strategy, max_tokens=256):
    messages = [
        {"role": "system", "content": "You are an expert tutor."},
        {"role": "user",   "content": example["prompt"]},
        {"role": "assistant", "content": "Let's think step by step."},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 1) Generate chain-of-thought
    cot_ids = generate_with_strategy(model, tokenizer, input_ids,
                                     strategy, max_tokens)
    # 2) Append delimiter and generate short answer
    answer_prompt = tokenizer("\n\nAnswer:", return_tensors="pt")["input_ids"].to(model.device)
    input_ids = torch.cat([input_ids, cot_ids.unsqueeze(0), answer_prompt], dim=-1)
    answer_ids = generate_with_strategy(model, tokenizer, input_ids,
                                        strategy, 16)

    cot_text = tokenizer.decode(cot_ids, skip_special_tokens=True)
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return cot_text, answer_text

class DecodingMethodEvaluator:
    def __init__(self, config, inject_reasoning=False, no_cache=False):
        self.config = config
        self.inject_reasoning = inject_reasoning
        self.no_cache = no_cache
        
        # Load model and tokenizer
        print(f"Loading model: {config['model']['name']}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            torch_dtype=torch.float16,
            device_map="auto",
            use_cache=not no_cache
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load data
        self.data = self.load_data(config['data']['input_file'])
        
        # Create results directory
        os.makedirs(os.path.dirname(config['data']['output_file']), exist_ok=True)
        
        # Initialize results DataFrame
        self.results = []
    
    def load_data(self, input_file):
        print(f"Loading data from: {input_file}")
        data = []
        with open(input_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        print(f"Loaded {len(data)} examples")
        return data
    
    def extract_answer_from_text(self, output_text, choices):
        """Extract answer key from generated output"""
        for label in choices['label']:
            if f" {label} " in output_text or f". {label} " in output_text or output_text.startswith(label):
                return label
        
        # If no answer found, return None
        return None
    
    def evaluate(self, strategy):
        """Run evaluation for all examples"""
        start_time = time.time()
        
        correct = 0
        total = 0
        results = []
        
        for example in tqdm(self.data):
            # Process example
            cot_text, answer_text = process_example(
                example, 
                self.model, 
                self.tokenizer, 
                strategy,
                max_tokens=self.config['model']['max_length']
            )
            
            # Extract the answer from the generated text
            predicted_answer = self.extract_answer_from_text(answer_text, example['choices'])
            
            # Check if correct
            if predicted_answer == example['answer_key']:
                correct += 1
            
            total += 1
            
            # Save detailed result
            results.append({
                "prompt": example['prompt'],
                "cot": cot_text,
                "answer": answer_text,
                "predicted": predicted_answer,
                "correct": predicted_answer == example['answer_key']
            })
        
        # Calculate accuracy and latency
        accuracy = correct / total if total > 0 else 0
        elapsed_time = time.time() - start_time
        avg_latency = elapsed_time / total if total > 0 else 0
        
        # Save detailed results
        results_dir = os.path.dirname(self.config['data']['output_file'])
        with open(os.path.join(results_dir, "detailed_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary = {
            "accuracy": accuracy,
            "latency": avg_latency,
            "total_time": elapsed_time,
            "total_examples": total,
            "correct": correct
        }
        
        with open(os.path.join(results_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Evaluation complete: Accuracy = {accuracy:.4f}, Avg Latency = {avg_latency:.4f} seconds")
        
        return accuracy, avg_latency, results

def main():
    parser = argparse.ArgumentParser(description="Run ARC-Easy evaluation with Deterministic Category-Based Sampling")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--inject_reasoning", action="store_true", help="Inject reasoning prefix")
    parser.add_argument("--no-cache", action="store_true", help="Disable KV-cache for benchmarking")
    parser.add_argument("--check-cache", action="store_true", help="Check cache speedup")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = DecodingMethodEvaluator(config, inject_reasoning=args.inject_reasoning, no_cache=args.no_cache)
    
    # Load cluster IDs
    try:
        # Try to load precomputed cluster IDs
        cluster_ids = torch.load("data/cluster_ids.pt", map_location="cpu")
        print("Loaded precomputed cluster IDs")
    except FileNotFoundError:
        print("No precomputed cluster IDs found, computing simple clusters")
        # Create a simple cluster assignment
        vocab_size = len(evaluator.tokenizer)
        cluster_ids = torch.zeros(vocab_size, dtype=torch.long)
        
        # Simple clustering by first character
        for i in range(vocab_size):
            token = evaluator.tokenizer.decode([i])
            if token and len(token) > 0:
                cluster_ids[i] = ord(token[0]) % 100
            else:
                cluster_ids[i] = 0
    
    # Create strategy
    strategy = DeterministicCategoryStrategy(cluster_ids)
    
    # Check cache speedup if requested
    if args.check_cache:
        # Run with cache first (default)
        print("Running benchmark with KV-cache...")
        evaluator_with_cache = DecodingMethodEvaluator(config, inject_reasoning=args.inject_reasoning, no_cache=False)
        # Only run on a small subset for timing comparison
        small_data = evaluator_with_cache.data[:5]
        evaluator_with_cache.data = small_data
        start_time = time.time()
        evaluator_with_cache.evaluate(strategy)
        cache_time = time.time() - start_time
        
        print("Running benchmark without KV-cache...")
        evaluator_no_cache = DecodingMethodEvaluator(config, inject_reasoning=args.inject_reasoning, no_cache=True)
        evaluator_no_cache.data = small_data
        start_time = time.time()
        evaluator_no_cache.evaluate(strategy)
        no_cache_time = time.time() - start_time
        
        # Compare latencies
        if cache_time > 0 and no_cache_time > 0:
            speedup = (no_cache_time - cache_time) / no_cache_time * 100
            print(f"Cache speed-up: {speedup:.2f}%")
            if speedup < 5:
                print(f"WARNING: KV-cache only provides {speedup:.2f}% speedup. Consider using --no-cache for simpler implementation.")
    
    # Run full evaluation
    evaluator.evaluate(strategy)

if __name__ == "__main__":
    main() 