# macOS Binaries

## Apple Silicon (M1/M2/M3/M4)

Optimized Metal backend for maximum performance on Apple GPUs.

### Tested Devices
- Apple M3 Pro
- Apple M1/M2 (community validated)
- Apple M4 (expected compatible)

### Requirements
- macOS 12.0+ (Monterey or later)
- ~2GB free storage
- Apple Silicon Mac

### Installation
```bash
curl -L https://github.com/akshaypn/qvac-finetune/releases/download/v1.0/qvac-macos-apple-silicon-v1.0.zip -o qvac-macos-apple-silicon-v1.0.zip
unzip qvac-macos-apple-silicon-v1.0.zip
cd qvac-macos-apple-silicon-v1.0
chmod +x bin/*
```

### Quick Start
```bash
# Download model
mkdir -p models
wget https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/qwen3-1_7b-q8_0.gguf -O models/qwen3-1.7b-q8_0.gguf

# Test inference
./bin/llama-cli -m models/qwen3-1.7b-q8_0.gguf -ngl 999 -p "Hello, world!"

# Download and extract biomedical dataset
wget https://github.com/tetherto/qvac-rnd-fabric-llm-finetune/raw/main/evaluation/biomedical_qa/biomedical_qa.zip
unzip biomedical_qa.zip

./bin/llama-finetune-lora \
  -m models/qwen3-1.7b-q8_0.gguf \
  -f biomedical_qa/train.jsonl \
  --assistant-loss-only \
  -c 128 -b 128 -ub 128 -ngl 999 -fa off \
  --learning-rate 1e-5 --lr-min 1e-8 \
  --lr-scheduler cosine --warmup-ratio 0.1 \
  --num-epochs 8 \
  --lora-modules "attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down"
```

### Performance (M3 Pro)

#### Inference
- **Qwen3-0.6B Q8**: ~90-120 tokens/sec
- **Qwen3-1.7B Q8**: ~62-90 tokens/sec
- **Gemma-1B Q8**: ~70-90 tokens/sec

#### Training
- **Qwen3-1.7B Q8**: ~17 tokens/sec
- **Time per epoch**: ~40 minutes
- **Full training (8 epochs)**: ~5-6 hours

### Using Trained Adapters
```bash
./bin/llama-cli \
  -m models/qwen3-1.7b-q8_0.gguf \
  --lora lora-adapter.gguf \
  -ngl 999 \
  -p "Your prompt here"
```

---

## Intel (x64)

For older Intel-based Macs (CPU-only).

### Requirements
- macOS 11.0+ (Big Sur or later)
- Intel-based Mac
- ~2GB free storage

### Installation
```bash
curl -L https://github.com/akshaypn/qvac-finetune/releases/download/v1.0/qvac-macos-intel-v1.0.zip -o qvac-macos-intel-v1.0.zip
unzip qvac-macos-intel-v1.0.zip
cd qvac-macos-intel-v1.0
chmod +x bin/*
```

### Performance Note
Intel Macs use CPU-only inference and training, which is significantly slower than Apple Silicon. Consider using smaller models or cloud resources for intensive workloads.

### Quick Start
```bash
# Use smaller models for CPU
mkdir -p models
wget https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/qwen3-0_6b-q4_0.gguf -O models/qwen3-0.6b-q4_0.gguf

# Inference (no -ngl flag for CPU)
./bin/llama-cli -m models/qwen3-0.6b-q4_0.gguf -p "Hello!"

# Fine-tuning (use minimal settings)
./bin/llama-finetune-lora \
  -m models/qwen3-0.6b-q4_0.gguf \
  -f dataset.jsonl \
  -c 64 -b 32 -ub 32 \
  --lora-rank 4 \
  --num-epochs 2
```

---

## Common Tasks

### Full LoRA Configuration
```bash
./bin/llama-finetune-lora \
  -m models/model.gguf \
  -f dataset.jsonl \
  --assistant-loss-only \
  -c 512 -b 128 -ub 128 -ngl 999 -fa off \
  --lora-rank 16 --lora-alpha 32 \
  --lora-modules "attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down" \
  --learning-rate 1e-5 --lr-min 1e-8 \
  --lr-scheduler cosine --warmup-ratio 0.1 \
  --checkpoint-save-steps 50 \
  --checkpoint-save-dir "./lora_checkpoints" \
  --num-epochs 8
```

### Resume from Checkpoint
```bash
./bin/llama-finetune-lora \
  -m models/model.gguf \
  -f dataset.jsonl \
  --resume-from "./lora_checkpoints/checkpoint_step_00000150/" \
  --output-adapter improved_adapter.gguf \
  -ngl 999
```

### Custom Chat Template
```bash
# Create custom template file (custom_template.jinja)
./bin/llama-finetune-lora \
  -m models/model.gguf \
  -f conversations.jsonl \
  --assistant-loss-only \
  --chat-template custom_template.jinja \
  -ngl 999
```

## Troubleshooting

### Permission Errors
```bash
chmod +x bin/*
```

### Metal Backend Not Found
```bash
# Verify you're using Apple Silicon binary
file bin/llama-cli
# Should show: Mach-O 64-bit executable arm64
```

### Out of Memory
```bash
# Reduce context length
-c 256  # or -c 128

# Use lower quantization
# Download Q4_0 instead of Q8_0

# Reduce LoRA rank
--lora-rank 8  # instead of 16
```

### Slow Performance
```bash
# Ensure GPU layers are enabled (Apple Silicon only)
-ngl 999

# Check system isn't throttling
# Monitor Activity Monitor > GPU History

# Close other GPU-intensive apps
```

## Model Recommendations

### Apple Silicon (M1/M2/M3)
- ✅ Qwen3-4B Q8_0 (8GB+ unified memory)
- ✅ Qwen3-1.7B Q8_0 (recommended)
- ✅ Gemma-1B Q8_0
- ✅ All 0.6B models

### Intel Macs
- ✅ Qwen3-0.6B Q4_0 (best performance)
- ⚠️ Qwen3-1.7B Q4_0 (very slow training)
- ❌ Models > 2B (impractical)

## Dataset Format

### JSON Lines (Instruction Fine-tuning)
```json
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
{"messages": [{"role": "user", "content": "What's 2+2?"}, {"role": "assistant", "content": "4"}]}
```

### Plain Text (Next-token Prediction)
```
This is plain text for training.
Each line is processed sequentially.
```

## Advanced Features

### Merge LoRA with Base Model
```bash
# Export merged model
./bin/llama-export-lora \
  -m models/base_model.gguf \
  --lora lora-adapter.gguf \
  -o merged_model.gguf
```

### Quantize Custom Model
```bash
./bin/llama-quantize \
  input_model.gguf \
  output_model_q8_0.gguf \
  Q8_0
```

## Performance Monitoring

### Activity Monitor
- Open Activity Monitor
- View > GPU History
- Monitor GPU usage during training

### Command-line
```bash
# Check system info
system_profiler SPHardwareDataType

# Monitor in real-time
sudo powermetrics --samplers gpu_power -i 1000
```

## Support

- 📚 [Main Documentation](../../README.md)
- 📊 [Benchmarks](../../docs/BENCHMARKS.md)
- 🐛 [Report Issues](https://github.com/akshaypn/qvac-finetune/issues)
- 💬 [Discussions](https://github.com/akshaypn/qvac-finetune/discussions)

## License

MIT License - See [LICENSE](../../LICENSE) for details.

