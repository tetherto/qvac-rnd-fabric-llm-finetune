# Model Inference Comparison Scripts

These scripts help you compare the performance of your fine-tuned LoRA adapters against the base Qwen model to evaluate if fine-tuning worked effectively.

## Scripts Overview

### 1. `compare_base_vs_adapters.py` - Comprehensive Comparison
Full comparison script that tests multiple questions across all models and saves detailed results.

**Features:**
- Tests base model + all LoRA adapters from your run
- Uses 5 default PubMedQA-style questions
- Saves detailed CSV results with metrics
- Provides summary statistics
- Supports custom questions

**Usage:**
```bash
# Basic usage with default questions
python compare_base_vs_adapters.py

# Specify custom run directory
python compare_base_vs_adapters.py --run_dir /path/to/your/run

# Use custom questions
python compare_base_vs_adapters.py --custom_questions "Does aspirin prevent heart attacks?" "Is exercise good for diabetes?"

# Adjust generation parameters
python compare_base_vs_adapters.py --max_new_tokens 300 --temperature 0.8
```

### 2. `quick_compare.py` - Single Question Test
Quick script for testing one question across all models. Great for rapid iteration.

**Usage:**
```bash
# Test a specific question
python quick_compare.py --question "Does vitamin D supplementation reduce respiratory infections?"

# Adjust max tokens
python quick_compare.py --question "Is metformin effective for PCOS?" --max_tokens 200
```

## Your Run Details

**Run Directory:** `/root/qvac-model-tools/qwen3-finetune/runs/20250911_084818`

**Available Adapters:**
- `run_gpu0_lora_r4` (LoRA rank 4)
- `run_gpu1_lora_r8` (LoRA rank 8) 
- `run_gpu2_lora_r16` (LoRA rank 16)
- `run_gpu3_lora_r24` (LoRA rank 24)
- `run_gpu4_lora_r32` (LoRA rank 32)
- `run_gpu5_lora_r40` (LoRA rank 40)
- `run_gpu6_lora_r64` (LoRA rank 64)
- `run_gpu7_lora_r128` (LoRA rank 128)

## Quick Start Examples

### Test if fine-tuning worked:
```bash
# Quick test with one question
python quick_compare.py --question "Can probiotics prevent antibiotic-associated diarrhea?"

# Full comparison with all default questions
python compare_base_vs_adapters.py
```

### Expected Results

**If fine-tuning worked well, you should see:**
- ✅ Longer, more detailed responses from fine-tuned models
- ✅ More medically accurate and structured answers
- ✅ Responses that follow the PubMedQA format better
- ✅ Different responses between base and fine-tuned models

**If fine-tuning didn't work:**
- ❌ Similar responses between base and fine-tuned models
- ❌ No improvement in medical accuracy or detail
- ❌ Responses still generic/not domain-specific

## Output Files

The comprehensive script creates:
- `comparison_results_TIMESTAMP.csv` - Detailed results for all models and questions
- Console output with side-by-side comparisons
- Summary statistics (response length, generation time)

## Tips for Analysis

1. **Response Quality:** Look for more detailed, medically accurate answers from fine-tuned models
2. **Consistency:** Check if higher LoRA ranks (r64, r128) perform better than lower ranks (r4, r8)
3. **Specialization:** Fine-tuned models should be better at medical question-answering format
4. **Length:** Fine-tuned models often generate longer, more comprehensive answers

## Troubleshooting

- **GPU Memory Issues:** The script loads models sequentially and cleans up memory between tests
- **Generation Speed:** Uses 8-bit quantization to balance memory usage and speed
- **Adapter Loading Errors:** Check that adapter paths exist and are properly saved

Run the scripts to see if your fine-tuning improved the model's medical question-answering capabilities!
