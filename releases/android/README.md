# Android Binaries

## Tested Devices
- Qualcomm Adreno 830 (Snapdragon 8 Elite)
- ARM Mali G715

## Requirements
- Android 10+ (API level 29+)
- Termux or similar terminal emulator
- ~4GB free storage
- Vulkan support (available on most modern Android devices)

## Installation

1. **Download the archive:**
```bash
wget https://github.com/akshaypn/qvac-finetune/releases/download/v1.0/qvac-android-adreno-arm64-v1.0.zip
unzip qvac-android-adreno-arm64-v1.0.zip
cd qvac-android-adreno-arm64-v1.0
```

2. **Set library path (REQUIRED for Android):**
```bash
export LD_LIBRARY_PATH=.
# Add to ~/.bashrc to make permanent
echo 'export LD_LIBRARY_PATH=.' >> ~/.bashrc
```

3. **Make binaries executable:**
```bash
chmod +x bin/*
```

4. **Test installation:**
```bash
./bin/llama-cli --version
```

## Quick Start

### Download a Model

Qwen3-0.6B is recommended for mobile devices:

```bash
mkdir -p models
wget https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/qwen3-0_6b-q8_0.gguf -O models/qwen3-0.6b-q8_0.gguf
```

### Run Inference

```bash
./bin/llama-cli -m models/qwen3-0.6b-q8_0.gguf -ngl 99 -p "Hello, what can you help me with?"
```

### Fine-tune with Biomedical Dataset

```bash
# Download biomedical dataset (included in repo)
wget https://raw.githubusercontent.com/akshaypn/qvac-finetune/main/datasets/biomedical_qa.jsonl

# Run LoRA fine-tuning
./bin/llama-finetune-lora \
  -m models/qwen3-0.6b-q8_0.gguf \
  -f biomedical_qa.jsonl \
  --assistant-loss-only \
  -c 128 -b 128 -ub 128 -ngl 99 -fa off \
  --num-epochs 2 \
  --lora-rank 8 --lora-alpha 16
```

### Use Trained Adapter

```bash
./bin/llama-cli \
  -m models/qwen3-0.6b-q8_0.gguf \
  --lora lora-adapter.gguf \
  -ngl 99 \
  -p "Does vitamin D supplementation prevent fractures?"
```

## Performance Expectations

### Qwen3-0.6B Q8_0 (Adreno 830)
- **Inference**: ~35-48 tokens/sec
- **Training**: ~6 tokens/sec
- **Training Time** (2 epochs): ~30-40 minutes

### Qwen3-1.7B Q8_0 (Adreno 830)
- **Inference**: ~17-24 tokens/sec
- **Training**: ~6 tokens/sec
- **Training Time** (8 epochs): ~1.5-2 hours

### Mali G715
- **Qwen3-0.6B**: ~15 tokens/sec inference
- **Qwen3-1.7B**: ~7-8 tokens/sec inference
- **Training**: Slower than Adreno but functional

## Tips for Mobile Training

### Memory Management
```bash
# Use smaller context for memory-constrained devices
-c 64    # Instead of 128

# Use Q4_0 quantization for larger models
wget https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/qwen3-1_7b-q4_0.gguf
```

### Battery Optimization
```bash
# Reduce batch size for cooler operation
-b 64 -ub 64

# Train in shorter sessions with checkpointing
--checkpoint-save-steps 25 --checkpoint-save-dir "./checkpoints"
```

### Keep Device Awake
```bash
# In Termux, prevent sleep
termux-wake-lock
```

## Troubleshooting

### Permission denied
```bash
chmod +x bin/*
```

### Library not found
```bash
# Make sure library path is set
export LD_LIBRARY_PATH=.
# Check it's set correctly
echo $LD_LIBRARY_PATH
```

### Out of memory (OOM)
- Use Q4_0 models instead of Q8_0
- Reduce context: `-c 64` or `-c 32`
- Use smaller models (0.6B instead of 1.7B)
- Close other apps

### Slow training
- Verify GPU is being used: `-ngl 99`
- Check Vulkan drivers: `vulkaninfo | grep GPU`
- Ensure device isn't thermal throttling (take a break to cool down)

### Device freezes
- Reduce batch size: `-b 32 -ub 32`
- Use checkpointing to save progress frequently
- Monitor temperature with a separate terminal

## Advanced Configuration

### Custom LoRA Parameters
```bash
./bin/llama-finetune-lora \
  -m models/qwen3-1.7b-q8_0.gguf \
  -f dataset.jsonl \
  --assistant-loss-only \
  -c 128 -b 128 -ub 128 -ngl 99 -fa off \
  --lora-rank 16 --lora-alpha 32 \
  --lora-modules "attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down" \
  --learning-rate 1e-5 --lr-min 1e-8 \
  --lr-scheduler cosine --warmup-ratio 0.1 \
  --num-epochs 8
```

### Learning Rate Scheduling
- **Cosine**: Smooth decay (recommended)
- **Linear**: Linear decay
- **Constant**: Fixed learning rate

### Checkpointing
```bash
# Auto-save every 50 steps
--checkpoint-save-steps 50 --checkpoint-save-dir "./lora_checkpoints"

# Resume from checkpoint
--resume-from "./lora_checkpoints/checkpoint_step_00000100/"
```

## Model Recommendations by Device

### Flagship Phones (Snapdragon 8 Gen 2/3, Dimensity 9200+)
- ✅ Qwen3-1.7B Q8_0 (best quality)
- ✅ Gemma-1B Q8_0
- ⚠️ Qwen3-4B Q4_0 (may OOM)

### Mid-range Phones
- ✅ Qwen3-0.6B Q8_0 (recommended)
- ✅ Qwen3-1.7B Q4_0
- ❌ Models > 2B (will OOM)

### Budget Phones
- ✅ Qwen3-0.6B Q4_0
- ⚠️ Qwen3-1.7B Q4_0 (may be slow)

## Dataset Format

### Instruction Fine-tuning (ChatML)
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

### Simple Text
```
This is plain text for next-token prediction fine-tuning.
Multiple lines are supported.
```

## Support

- 📚 [Main Documentation](../../README.md)
- 🐛 [Report Issues](https://github.com/akshaypn/qvac-finetune/issues)
- 💬 [Discussions](https://github.com/akshaypn/qvac-finetune/discussions)

## License

MIT License - See [LICENSE](../../LICENSE) for details.

