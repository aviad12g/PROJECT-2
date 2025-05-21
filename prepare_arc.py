#!/usr/bin/env python
# Script to prepare ARC-Easy examples for evaluation with Deterministic Category-Based Sampling

import json
import os
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("Loading ARC-Easy dev dataset...")
    # Load ARC-Easy dataset from Hugging Face
    dataset = load_dataset("ai2_arc", "ARC-Easy", split="validation")
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Process each example and format for chat template use later
    processed_examples = []
    
    for example in tqdm(dataset):
        question = example["question"]
        choices = example["choices"]
        answer_key = example["answerKey"]
        
        # Format choices as A. choice1, B. choice2, etc.
        formatted_choices = []
        for i, choice in enumerate(choices["text"]):
            label = choices["label"][i]
            formatted_choices.append(f"{label}. {choice}")
        
        # Create prompt (without chain-of-thought - we'll add that in chat template)
        prompt = f"Question: {question}\n\nChoices:\n" + "\n".join(formatted_choices)
        
        # Store the example
        processed_examples.append({
            "id": example["id"],
            "prompt": prompt,
            "choices": choices,
            "answer_key": answer_key
        })
    
    # Save to JSONL file
    output_file = "results/arc_easy_dev.jsonl"
    print(f"Saving {len(processed_examples)} examples to {output_file}...")
    
    with open(output_file, "w") as f:
        for example in processed_examples:
            f.write(json.dumps(example) + "\n")
    
    print("Done!")
    
    # Also copy to GCS if environment variables are set
    bucket_name = os.environ.get("GCS_BUCKET", "arc-easy-bucket")
    if bucket_name:
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob("arc_easy_dev.jsonl")
            blob.upload_from_filename(output_file)
            print(f"Uploaded to gs://{bucket_name}/arc_easy_dev.jsonl")
        except Exception as e:
            print(f"Warning: Failed to upload to GCS: {e}")

if __name__ == "__main__":
    main() 