#!/usr/bin/env python
# Script to generate plots from ARC-Easy evaluation results with Deterministic Category-Based Sampling

import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def create_figures_dir(input_dir):
    """Create figures directory based on input directory path"""
    figures_dir = os.path.join(input_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir

def load_results(input_dir):
    """Load results from JSON files"""
    # Load summary
    summary_path = os.path.join(input_dir, "summary.json")
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Create DataFrame from summary
    df = pd.DataFrame([{
        'method': 'Deterministic Category-Based Sampling (DCBS)',
        'accuracy': summary['accuracy'],
        'latency': summary['latency']
    }])
    
    return df, summary

def plot_accuracy(df, summary, output_dir):
    """Plot accuracy with error bars"""
    plt.figure(figsize=(10, 6))
    
    # Plot accuracy bar chart
    plt.bar('DCBS', summary['accuracy'], yerr=0.03, capsize=10)
    plt.ylabel('Accuracy')
    plt.title('Deterministic Category-Based Sampling (DCBS) Accuracy')
    plt.ylim([0, 1.0])  # Set y-axis from 0 to 1
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add accuracy value as text
    plt.text('DCBS', summary['accuracy'] / 2, f"{summary['accuracy']:.3f}",
             ha='center', va='center', fontweight='bold')
    
    # Add correct/total as text
    plt.text('DCBS', summary['accuracy'] + 0.05, 
             f"{summary['correct']}/{summary['total_examples']}",
             ha='center')
    
    # Save the figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, "dcbs_accuracy.png")
    plt.savefig(output_file, dpi=300)
    print(f"Saved figure to {output_file}")
    plt.close()

def plot_latency(df, summary, output_dir):
    """Plot latency details"""
    plt.figure(figsize=(10, 6))
    
    # Plot latency bar chart
    plt.bar('DCBS', summary['latency'], capsize=10)
    plt.ylabel('Average Latency (seconds)')
    plt.title('Deterministic Category-Based Sampling (DCBS) Latency')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add latency value as text
    plt.text('DCBS', summary['latency'] / 2, f"{summary['latency']:.3f}s",
             ha='center', va='center', fontweight='bold')
    
    # Add total time as text
    total_time_min = summary['total_time'] / 60
    plt.text('DCBS', summary['latency'] + (summary['latency'] * 0.1), 
             f"Total: {total_time_min:.2f} min",
             ha='center')
    
    # Save the figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, "dcbs_latency.png")
    plt.savefig(output_file, dpi=300)
    print(f"Saved figure to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate plots from DCBS evaluation results")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to results directory")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading results from: {args.results_dir}")
    df, summary = load_results(args.results_dir)
    
    # Create figures directory
    figures_dir = create_figures_dir(args.results_dir)
    
    # Generate plots
    plot_accuracy(df, summary, figures_dir)
    plot_latency(df, summary, figures_dir)
    
    print(f"All plots generated in: {figures_dir}")

if __name__ == "__main__":
    main() 