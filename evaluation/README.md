# Evaluation & Results

This directory contains datasets, evaluation scripts, and comprehensive reports from fine-tuning experiments across heterogeneous GPU platforms.



## 📊 Datasets

### 1. Biomedical Question-Answering

**Location:** `./biomedical_qa/biomedical_qa.zip`

**Purpose:** Domain-specific instruction fine-tuning for medical/scientific knowledge

**Statistics:**
- **Total**: 330 examples
- **Train**: 264 examples
- **Validation**: 33 examples
- **Test**: 33 examples
- **Classes**: Yes (110), No (110), Uncertain (110) - balanced
- **Source**: PubMedQA (MIT License)

**Format:**
```json
{
  "text": "Q: Does vitamin D prevent fractures?\nA: Yes. Rationale: Meta-analysis shows...",
  "label": "Yes",
  "question": "Does vitamin D prevent fractures?",
  "rationale_src": "Meta-analysis shows significant reduction in hip fractures...",
  "source": "pqa_labeled"
}
```

**Validated Results:**
- **Mobile GPUs** (Mali/Adreno): 79-81% accuracy
- **Desktop GPUs**: 82-94% accuracy
- **vs PyTorch**: Competitive (78-86% PyTorch baseline)

**Reports:**
- [Biomedical Fine-tuning Report](./reports/BIOMED_FINETUNING_REPORT.md)
- [Complete Project Report](./reports/COMPLETE_PROJECT_REPORT.md)

---

### 2. Email Style Transfer

**Location:** `./email_style_transfer/`

**Purpose:** Unstructured text fine-tuning for personal writing style adaptation

**Statistics:**
- **Total**: 200 emails
- **Format**: Synthetic, privacy-safe personal emails
- **Use Case**: Style transfer, casual communication patterns

**Format:**
```json
{
  "id": 1,
  "subject": "re: quick health update!! 🌙 ..",
  "body": "hey jordan,\n\nforwarding the receipt so we can split later..."
}
```

**Characteristics:**
- Casual, conversational tone
- Varied subject lines (emojis, punctuation)
- Short, practical messages
- Personal yet generic (no PII)

**Use Case:**
Demonstrates next-token prediction fine-tuning for style adaptation. Models learn to mimic specific writing patterns, tone, and formatting preferences.

---

## 🔬 Evaluation Scripts

### compare_base_vs_adapters.py

Compares base model performance against fine-tuned LoRA adapters.

**Usage:**
```bash
python scripts/compare_base_vs_adapters.py \
  --base-model models/base.gguf \
  --adapter models/lora_adapter.gguf \
  --test-data biomedical_qa/test.jsonl \
  --output results.json
```

**Metrics:**
- Accuracy on test set
- Cosine similarity
- Jaccard similarity
- Perplexity

---

### test_biomed_prompts.py

Tests fine-tuned models on biomedical prompts with ground truth validation.

**Usage:**
```bash
python scripts/test_biomed_prompts.py \
  --model models/base.gguf \
  --adapter models/lora_adapter.gguf \
  --dataset biomedical_qa/test.jsonl \
  --num-samples 100
```

**Output:**
- Per-sample predictions
- Accuracy metrics
- Confusion matrix
- Error analysis

---

### monitor_training.py

Real-time training monitoring and logging.

**Usage:**
```bash
python scripts/monitor_training.py \
  --log-dir ./training_logs \
  --output-csv training_metrics.csv
```

**Monitors:**
- Loss over time
- Learning rate schedule
- Gradient norms
- Memory usage
- Tokens/second

---

### quick_compare.py

Quick comparison tool for rapid iteration testing.

**Usage:**
```bash
python scripts/quick_compare.py \
  --model-a models/adapter_v1.gguf \
  --model-b models/adapter_v2.gguf \
  --test-prompts test_prompts.txt
```

**Output:**
- Side-by-side responses
- Quality scores
- Response time comparison

---

## 📄 Comprehensive Reports

### BIOMED_FINETUNING_REPORT.md

Detailed report on biomedical dataset fine-tuning experiments.

**Contents:**
- Dataset creation methodology
- Training configuration
- Cross-platform results
- Quality metrics
- Comparison with PyTorch/HuggingFace

**Key Findings:**
- ✅ 79-94% accuracy across all platforms
- ✅ Near-parity with PyTorch (78-86%)
- ✅ Mobile GPUs achieve clinical-grade accuracy

---

### COMPLETE_PROJECT_REPORT.md

Comprehensive documentation of the entire fine-tuning project.

**Contents:**
- Project overview and motivation
- Technical implementation details
- Cross-platform validation
- Performance benchmarks
- Future work and recommendations

---

### README_inference_comparison.md

Inference performance comparison across platforms.

**Contents:**
- Inference speed metrics
- TTFT (Time To First Token)
- Throughput measurements
- Memory usage analysis

---

### biomed_comparison_results.json

Raw numerical results from biomedical experiments.

**Structure:**
```json
{
  "model": "qwen3-1.7b-q8_0",
  "platform": "adreno-830",
  "accuracy": 0.8143,
  "loss": 0.234,
  "training_time": "1h 40m",
  "tokens_per_second": 6.09
}
```

---

## 🚀 Quick Start

### Run Biomedical Evaluation

```bash
# Extract biomedical dataset
cd evaluation/biomedical_qa
unzip biomedical_qa.zip
cd ../..

# Fine-tune on biomedical dataset
./bin/llama-finetune-lora \
  -m models/qwen3-1.7b-q8_0.gguf \
  -f evaluation/biomedical_qa/biomedical_qa/train.jsonl \
  --assistant-loss-only \
  -c 128 -b 128 -ub 128 -ngl 999 -fa off \
  --num-epochs 8 \
  --output-adapter biomedical_adapter.gguf

# Test on validation set
python evaluation/scripts/test_biomed_prompts.py \
  --model models/qwen3-1.7b-q8_0.gguf \
  --adapter biomedical_adapter.gguf \
  --dataset evaluation/biomedical_qa/biomedical_qa/validation.jsonl
```

### Run Email Style Transfer

```bash
# Fine-tune on email dataset
./bin/llama-finetune-lora \
  -m models/qwen3-1.7b-q8_0.gguf \
  -f evaluation/email_style_transfer/email_dataset.jsonl \
  -c 512 -b 128 -ub 128 -ngl 999 \
  --lora-rank 16 --lora-alpha 32 \
  --num-epochs 3 \
  --output-adapter email_adapter.gguf

# Test style transfer
./bin/llama-cli \
  -m models/qwen3-1.7b-q8_0.gguf \
  --lora email_adapter.gguf \
  -ngl 999 \
  -p "Write a quick email about meeting for coffee this weekend"
```

---

## 📊 Expected Results

### Biomedical Q&A

| Platform | Accuracy | Training Time (8 epochs) |
|----------|----------|--------------------------|
| **RTX 4090** | 86-94% | 45 min |
| **AMD 7900 XTX** | 82-92% | 1.7 hrs |
| **Apple M3 Pro** | 82-89% | 5.3 hrs |
| **Adreno 830** | 81-84% | 13 hrs |
| **Mali G715** | 79-82% | 61 hrs |

### Email Style Transfer

**Qualitative Results:**
- ✅ Learns casual tone and emoji usage
- ✅ Mimics subject line patterns
- ✅ Adopts personal sign-offs (A, Alex, - a)
- ✅ Uses similar formatting (bullets, quotes)

---

## 🔧 Evaluation Guidelines

### Creating New Evaluations

1. **Prepare Dataset**
   ```bash
   # Create test_dataset.jsonl
   # Format: one JSON object per line
   ```

2. **Run Fine-tuning**
   ```bash
   ./bin/llama-finetune-lora -m model.gguf -f test_dataset.jsonl [options]
   ```

3. **Evaluate Results**
   ```bash
   python evaluation/scripts/test_biomed_prompts.py --model model.gguf --adapter adapter.gguf
   ```

4. **Document Findings**
   ```bash
   # Create evaluation report in ./reports/
   ```

### Metrics to Track

**Quantitative:**
- Accuracy (classification tasks)
- Perplexity (language modeling)
- BLEU/ROUGE scores (generation)
- Cosine similarity (embeddings)
- Training time
- Tokens/second

**Qualitative:**
- Response quality
- Style adherence
- Factual correctness
- Formatting consistency

---

## 📝 Contributing Evaluations

We welcome evaluation contributions!

**Requirements:**
1. Clear dataset documentation
2. Reproducible training commands
3. Comprehensive metrics
4. Cross-platform testing (if possible)
5. Comparison with baseline

**Submission:**
1. Add dataset to appropriate subdirectory
2. Include evaluation scripts
3. Document results in report
4. Submit pull request

---

## 🔗 Related Documentation

- [Main README](../README.md) - Project overview
- [Benchmarks](../docs/BENCHMARKS.md) - Performance metrics

---

## 📄 License

- **Biomedical Dataset**: MIT (PubMedQA-derived)
- **Email Dataset**: Internal research use
- **Scripts**: MIT License

---

<div align="center">
  <p><b>Comprehensive evaluation across heterogeneous platforms</b></p>
  <p>Demonstrating universal fine-tuning quality</p>
</div>

