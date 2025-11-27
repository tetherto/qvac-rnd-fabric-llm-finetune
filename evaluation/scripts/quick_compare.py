#!/usr/bin/env python3
"""
Quick comparison script for testing a single question across base model and all adapters.
Useful for rapid testing during development.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
import argparse

def load_base_model(model_id="Qwen/Qwen3-1.7B-Base"):
    """Load base model and tokenizer."""
    print(f"Loading base model: {model_id}")
    
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

@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens=150):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Use greedy decoding for consistency
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    return generated_text.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="Question to ask all models")
    parser.add_argument("--run_dir", default="/root/qvac-model-tools/qwen3-finetune/runs/20250911_084818")
    parser.add_argument("--max_tokens", type=int, default=150)
    
    args = parser.parse_args()
    
    # Format question
    if not args.question.startswith("Question:"):
        question = f"Question: {args.question}\nAnswer:"
    else:
        question = args.question if args.question.endswith("Answer:") else f"{args.question}\nAnswer:"
    
    print("="*80)
    print(f"TESTING QUESTION: {args.question}")
    print("="*80)
    
    # Load base model
    base_model, tokenizer = load_base_model()
    
    # Test base model
    print(f"\n[BASE MODEL]")
    base_response = generate_response(base_model, tokenizer, question, args.max_tokens)
    print(f"Response: {base_response}")
    
    # Find and test all adapters
    run_path = Path(args.run_dir)
    adapter_dirs = sorted([d for d in run_path.iterdir() if d.is_dir() and d.name.startswith("run_gpu")])
    
    for adapter_dir in adapter_dirs:
        final_adapter = adapter_dir / "final_adapter"
        if final_adapter.exists():
            rank = adapter_dir.name.split("_r")[-1]
            print(f"\n[ADAPTER: {adapter_dir.name}] (rank={rank})")
            
            try:
                # Load adapter
                adapter_model = PeftModel.from_pretrained(base_model, str(final_adapter))
                
                # Generate response
                adapter_response = generate_response(adapter_model, tokenizer, question, args.max_tokens)
                print(f"Response: {adapter_response}")
                
                # Compare with base
                if adapter_response != base_response:
                    print("✓ DIFFERENT from base model")
                else:
                    print("⚠ SAME as base model")
                
                # Cleanup
                del adapter_model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error loading adapter: {e}")

if __name__ == "__main__":
    main()
