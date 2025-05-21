import argparse
import json
import os
import time
import yaml
import pandas as pd
from tqdm import tqdm
from vertexai.preview.language_models import ChatModel

from src.dcbs import DeterministicCategoryStrategy


def process_example(example, chat):
    prompt = f"{example['prompt']}\nLet's think step by step."
    cot_response = chat.send_message(prompt).text

    answer_prompt = cot_response + "\n\nAnswer:"
    answer_response = chat.send_message(answer_prompt).text

    return cot_response, answer_response.strip()


class DecodingMethodEvaluator:
    def __init__(self, config, inject_reasoning=False):
        self.config = config
        self.inject_reasoning = inject_reasoning

        model_id = config["model"]["name"]
        print(f"Loading Gemini model: {model_id}")
        chat_model = ChatModel.from_pretrained(model_id)
        self.chat = chat_model.start_chat()

        # Load data
        self.data = self.load_data(config["data"]["input_file"])

        # Create results directory
        os.makedirs(os.path.dirname(config["data"]["output_file"]), exist_ok=True)

    def load_data(self, input_file):
        print(f"Loading data from: {input_file}")
        data = []
        with open(input_file, "r") as f:
            for line in f:
                data.append(json.loads(line))
        print(f"Loaded {len(data)} examples")
        return data

    def extract_answer_from_text(self, output_text, choices):
        for label in choices["label"]:
            if (
                f" {label} " in output_text
                or f". {label} " in output_text
                or output_text.startswith(label)
            ):
                return label
        return None

    def evaluate(self, strategy):
        start_time = time.time()
        correct = 0
        total = 0
        results = []

        for example in tqdm(self.data):
            cot_text, answer_text = process_example(example, self.chat)
            predicted_answer = self.extract_answer_from_text(
                answer_text, example["choices"]
            )

            if predicted_answer == example["answer_key"]:
                correct += 1
            total += 1

            results.append(
                {
                    "prompt": example["prompt"],
                    "cot": cot_text,
                    "answer": answer_text,
                    "predicted": predicted_answer,
                    "correct": predicted_answer == example["answer_key"],
                }
            )

        accuracy = correct / total if total > 0 else 0
        elapsed_time = time.time() - start_time
        avg_latency = elapsed_time / total if total > 0 else 0

        results_dir = os.path.dirname(self.config["data"]["output_file"])
        with open(os.path.join(results_dir, "detailed_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        summary = {
            "accuracy": accuracy,
            "latency": avg_latency,
            "total_time": elapsed_time,
            "total_examples": total,
            "correct": correct,
        }
        with open(os.path.join(results_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(
            f"✅ Evaluation complete: Accuracy = {accuracy:.4f}, Avg Latency = {avg_latency:.4f} sec"
        )
        return accuracy, avg_latency, results


def main():
    parser = argparse.ArgumentParser(
        description="Run ARC-Easy evaluation with Gemini 2.0 Flash"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--inject_reasoning", action="store_true", help="Inject reasoning prefix"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    evaluator = DecodingMethodEvaluator(
        config, inject_reasoning=args.inject_reasoning
    )

    # Gemini doesn’t use token-level cluster logic; dummy strategy
    cluster_ids = {}
    strategy = DeterministicCategoryStrategy(cluster_ids)

    evaluator.evaluate(strategy)


if __name__ == "__main__":
    main()
