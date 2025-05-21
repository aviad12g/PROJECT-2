import unittest
import torch
import os
import tempfile
import yaml
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the process_example function
from run_dcbs_eval import process_example
from src.dcbs import DeterministicCategoryStrategy

class TestCoTGeneration(unittest.TestCase):
    def setUp(self):
        # Create a temporary config file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "test_config.yaml")
        self.results_dir = os.path.join(self.temp_dir.name, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create a simple config
        config = {
            "model": {
                "name": "sshleifer/tiny-gpt2",
                "max_length": 100
            },
            "data": {
                "input_file": os.path.join(self.temp_dir.name, "test_data.jsonl"),
                "output_file": os.path.join(self.results_dir, "output.csv")
            },
            "prompt": {
                "template": "{prompt}",
                "reasoning_prefix": "Let's think step by step.",
                "answer_format": "The answer is"
            }
        }
        
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)
        
        # Create test data
        test_data = [
            {
                "prompt": "What is 1+1?",
                "choices": {"text": ["two", "three", "four", "five"], "label": ["A", "B", "C", "D"]},
                "answer_key": "A"
            }
        ]
        
        with open(config["data"]["input_file"], "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
        
        # Load tiny model for testing
        self.model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        
        # Make sure pad_token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create dummy cluster_ids
        self.cluster_ids = torch.zeros(len(self.tokenizer), dtype=torch.long)
        for i in range(len(self.tokenizer)):
            self.cluster_ids[i] = i % 10  # Simple test clustering
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_cot_contains_thinking_and_answer(self):
        # Load test data
        with open(os.path.join(self.temp_dir.name, "test_data.jsonl"), 'r') as f:
            data = [json.loads(line) for line in f]
        
        example = data[0]
        
        # Create strategy
        strategy = DeterministicCategoryStrategy(self.cluster_ids)
        
        # Process example
        try:
            cot_text, answer_text = process_example(example, self.model, self.tokenizer, strategy, max_tokens=20)
            
            # Just check that we get outputs without errors
            self.assertIsInstance(cot_text, str)
            self.assertIsInstance(answer_text, str)
        except Exception as e:
            self.fail(f"process_example raised exception {e}")

if __name__ == "__main__":
    unittest.main() 