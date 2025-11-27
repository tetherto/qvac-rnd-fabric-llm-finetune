#!/usr/bin/env python3
"""
Compare inference results between base Qwen model and all fine-tuned LoRA adapters.
This script helps evaluate if fine-tuning worked by comparing responses to the same questions.
"""

import os
import torch
import argparse
from pathlib import Path
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import time
from datetime import datetime

# Sample PubMedQA-style questions to test
SAMPLE_QUESTIONS = [
    "Question: Does vitamin D supplementation reduce the risk of respiratory tract infections?\nAnswer:",
    "Question: Is metformin effective in treating polycystic ovary syndrome?\nAnswer:",
    "Question: Can probiotics prevent antibiotic-associated diarrhea?\nAnswer:",
    "Question: Does exercise training improve quality of life in patients with heart failure?\nAnswer:",
    "Question: Is cognitive behavioral therapy effective for treating chronic pain?\nAnswer:"
]

def get_base_model_and_tokenizer(model_id="Qwen/Qwen3-1.7B-Base"):
    """Load the base model and tokenizer."""
    print(f"Loading base model: {model_id}")
    
    # Use 8-bit quantization like in training
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_adapter_model(base_model, adapter_path):
    """Load a PEFT adapter on top of the base model."""
    print(f"Loading adapter: {adapter_path}")
    try:
        adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
        return adapter_model
    except Exception as e:
        print(f"Error loading adapter {adapter_path}: {e}")
        return None

@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7):
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generation_time = time.time() - start_time
    
    # Decode only the generated part (not the input prompt)
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    return generated_text.strip(), generation_time

def find_adapter_paths(run_dir):
    """Find all adapter paths in the run directory."""
    run_path = Path(run_dir)
    adapter_paths = []
    
    for subdir in sorted(run_path.iterdir()):
        if subdir.is_dir() and subdir.name.startswith("run_gpu"):
            final_adapter = subdir / "final_adapter"
            if final_adapter.exists():
                # Extract rank from directory name
                rank_str = subdir.name.split("_r")[-1]
                adapter_paths.append({
                    'path': str(final_adapter),
                    'name': subdir.name,
                    'rank': int(rank_str)
                })
    
    # Sort by rank
    adapter_paths.sort(key=lambda x: x['rank'])
    return adapter_paths

def main():
    parser = argparse.ArgumentParser(description="Compare base model with fine-tuned adapters")
    parser.add_argument("--run_dir", default="/root/qvac-model-tools/qwen3-finetune/runs/20250911_084818",
                        help="Directory containing the adapter runs")
    parser.add_argument("--model_id", default="Qwen/Qwen3-1.7B-Base", 
                        help="Base model ID")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    parser.add_argument("--output_file", default=None,
                        help="Output CSV file (default: comparison_results_TIMESTAMP.csv)")
    parser.add_argument("--custom_questions", nargs="+", default=None,
                        help="Custom questions to test (space-separated)")
    
    args = parser.parse_args()
    
    # Set up output file
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"comparison_results_{timestamp}.csv"
    
    print("="*80)
    print("QWEN3 BASE MODEL vs FINE-TUNED ADAPTERS COMPARISON")
    print("="*80)
    print(f"Run directory: {args.run_dir}")
    print(f"Base model: {args.model_id}")
    print(f"Output file: {args.output_file}")
    print()
    
    # Use custom questions if provided, otherwise use samples
    if args.custom_questions:
        questions = [f"Question: {q}\nAnswer:" for q in args.custom_questions]
    else:
        questions = SAMPLE_QUESTIONS
    
    print(f"Testing {len(questions)} questions")
    
    # Load base model
    base_model, tokenizer = get_base_model_and_tokenizer(args.model_id)
    
    # Find all adapter paths
    adapter_paths = find_adapter_paths(args.run_dir)
    print(f"Found {len(adapter_paths)} adapters:")
    for adapter_info in adapter_paths:
        print(f"  - {adapter_info['name']} (rank={adapter_info['rank']})")
    print()
    
    # Store all results
    all_results = []
    
    # Test each question
    for q_idx, question in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"QUESTION {q_idx + 1}: {question.split('Answer:')[0].replace('Question: ', '').strip()}")
        print(f"{'='*60}")
        
        # Generate response from base model
        print("\n[BASE MODEL]")
        base_response, base_time = generate_response(
            base_model, tokenizer, question, 
            args.max_new_tokens, args.temperature
        )
        print(f"Response: {base_response}")
        print(f"Generation time: {base_time:.2f}s")
        
        # Store base model result
        all_results.append({
            'question_idx': q_idx + 1,
            'question': question.split('Answer:')[0].replace('Question: ', '').strip(),
            'model_type': 'base',
            'model_name': 'base_model',
            'rank': 0,
            'response': base_response,
            'generation_time': base_time,
            'response_length': len(base_response.split())
        })
        
        # Test each adapter
        for adapter_info in adapter_paths:
            print(f"\n[{adapter_info['name'].upper()}] (rank={adapter_info['rank']})")
            
            # Load adapter
            adapter_model = load_adapter_model(base_model, adapter_info['path'])
            if adapter_model is None:
                continue
            
            # Generate response
            adapter_response, adapter_time = generate_response(
                adapter_model, tokenizer, question,
                args.max_new_tokens, args.temperature
            )
            print(f"Response: {adapter_response}")
            print(f"Generation time: {adapter_time:.2f}s")
            
            # Store adapter result
            all_results.append({
                'question_idx': q_idx + 1,
                'question': question.split('Answer:')[0].replace('Question: ', '').strip(),
                'model_type': 'adapter',
                'model_name': adapter_info['name'],
                'rank': adapter_info['rank'],
                'response': adapter_response,
                'generation_time': adapter_time,
                'response_length': len(adapter_response.split())
            })
            
            # Clean up adapter to save memory
            del adapter_model
            torch.cuda.empty_cache()
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output_file, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {args.output_file}")
    print(f"{'='*80}")
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 40)
    
    # Average response length by model
    avg_lengths = results_df.groupby(['model_type', 'model_name'])['response_length'].mean()
    print("\nAverage response length (words):")
    for (model_type, model_name), avg_length in avg_lengths.items():
        rank_info = f" (rank={results_df[results_df['model_name']==model_name]['rank'].iloc[0]})" if model_type == 'adapter' else ""
        print(f"  {model_name}{rank_info}: {avg_length:.1f}")
    
    # Average generation time by model
    avg_times = results_df.groupby(['model_type', 'model_name'])['generation_time'].mean()
    print("\nAverage generation time (seconds):")
    for (model_type, model_name), avg_time in avg_times.items():
        rank_info = f" (rank={results_df[results_df['model_name']==model_name]['rank'].iloc[0]})" if model_type == 'adapter' else ""
        print(f"  {model_name}{rank_info}: {avg_time:.2f}")
    
    print(f"\nDetailed results available in: {args.output_file}")
    print("\nTo analyze the results further, you can:")
    print("1. Open the CSV file to compare responses side-by-side")
    print("2. Look for differences in response quality and medical accuracy")
    print("3. Check if fine-tuned models provide more detailed/relevant answers")

if __name__ == "__main__":
    main()
