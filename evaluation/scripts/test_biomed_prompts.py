#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

def load_base_model():
    """Load the base Qwen-3 1.7B model"""
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B-Base",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_finetuned_model(adapter_path):
    """Load the fine-tuned model with LoRA adapter"""
    base_model, tokenizer = load_base_model()
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=64):
    """Generate response from model"""
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

def main():
    print("🧪 Testing Biomedical Yes/No Questions: Base vs Fine-tuned")
    print("=" * 80)
    
    # Test prompts
    test_prompts = [
        "Q: Does vitamin D supplementation reduce acute respiratory infection risk in adults?",
        "Q: Do antibiotics shorten illness in uncomplicated acute bronchitis?",
        "Q: Is a D-dimer test alone enough to rule out pulmonary embolism in high pretest probability patients?",
        "Q: Do statins reduce all-cause mortality for primary prevention in adults over 75?",
        "Q: Does annual low-dose CT screening reduce lung-cancer mortality in long-term smokers?",
        "Q: Do probiotics lower the risk of antibiotic-associated diarrhea in outpatients?",
        "Q: Does long-term PPI use increase the risk of Clostridioides difficile infection?",
        "Q: Do SGLT2 inhibitors increase euglycemic DKA risk in type 2 diabetes?",
        "Q: Should beta-blockers be used in acute cocaine-associated chest pain?",
        "Q: An RCT reports RR = 0.85 (95% CI 0.60–1.05). Is this statistically significant?"
    ]
    
    expected_base_responses = [
        "Vitamin D supplementation has been shown to reduce respiratory infections in adults. Trial registration: NCT00000000. PMID: 23456789.",
        "Yes, antibiotics reduce symptom duration and prevent complications like pneumonia in bronchitis. Clinicians often prescribe macrolides. PMID: 21861111.",
        "A negative D-dimer can effectively rule out PE in many patients when interpreted correctly.",
        "Yes, statins reduce cardiovascular events and overall mortality in older adults. DOI: 10.1000/j.journal.12345.",
        "Yes, annual LDCT lowers lung cancer mortality in high-risk smokers; see major trials and guidelines.",
        "Probiotics significantly reduce antibiotic-associated diarrhea; multiple meta-analyses confirm. PMID: 22398545.",
        "Yes, PPIs increase the risk of C. difficile infection; clinicians should deprescribe when possible.",
        "SGLT2 inhibitors improve glycemic control without meaningful DKA risk in most patients.",
        "Beta-blockers reduce myocardial oxygen demand and are appropriate in chest pain management.",
        "Yes, RR below 1.0 indicates a significant risk reduction compared with control."
    ]
    
    expected_finetuned_responses = [
        "Uncertain. Benefit mainly in vitamin-D-deficient groups; overall RCT results are mixed.",
        "No. Minimal or no benefit for uncomplicated cases; supportive care is preferred.",
        "No. In high pretest probability, imaging is required; D-dimer alone is insufficient.",
        "Uncertain. Evidence for mortality benefit is limited/mixed; individualize based on risk and frailty.",
        "Yes. Demonstrated mortality reduction in eligible high-risk groups under screening criteria.",
        "Yes. Modest relative risk reduction; product/strain and timing matter.",
        "Yes. Association observed; use lowest effective dose and reassess indication.",
        "Yes. Rare but increased risk; hold during acute illness, surgery, or prolonged fasting.",
        "No. Avoid acutely due to unopposed α-stimulation; treat with benzodiazepines/nitrates.",
        "No. The confidence interval crosses 1.0, so not statistically significant."
    ]
    
    # Load models
    print("Loading base model...")
    base_model, base_tokenizer = load_base_model()
    
    print("Loading fine-tuned model (LoRA r=16)...")
    adapter_path = "runs/biomed_yesno_20250915_150429/run_gpu2_lora_r16/final_adapter"
    try:
        finetuned_model, finetuned_tokenizer = load_finetuned_model(adapter_path)
        print("✅ Fine-tuned model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading fine-tuned model: {e}")
        print("Using base model for both comparisons")
        finetuned_model, finetuned_tokenizer = base_model, base_tokenizer
    
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}) {prompt}")
        print("-" * 60)
        
        # Generate base model response
        base_response = generate_response(base_model, base_tokenizer, prompt + "\nA:")
        print(f"BASE: {base_response}")
        
        # Generate fine-tuned model response  
        finetuned_response = generate_response(finetuned_model, finetuned_tokenizer, prompt + "\nA:")
        print(f"FINETUNED: {finetuned_response}")
        
        # Show expected responses for reference
        print(f"EXPECTED BASE: {expected_base_responses[i-1]}")
        print(f"EXPECTED FINETUNED: {expected_finetuned_responses[i-1]}")
        
        results.append({
            "question": prompt,
            "base_response": base_response,
            "finetuned_response": finetuned_response,
            "expected_base": expected_base_responses[i-1],
            "expected_finetuned": expected_finetuned_responses[i-1]
        })
    
    # Save results
    with open("biomed_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ Comparison complete! Results saved to 'biomed_comparison_results.json'")
    print("=" * 80)

if __name__ == "__main__":
    main()

