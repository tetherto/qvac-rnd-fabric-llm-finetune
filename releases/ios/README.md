# iOS Binaries

## Overview

Fine-tune and run LLMs directly on iPhone and iPad using the Metal backend.

### Tested Devices
- iPhone 15 Pro (A17 Pro)
- iPad Pro (M1/M2)
- Expected compatible with iPhone 13+ and iPad Air 5+

### Requirements
- iOS 16.0+ / iPadOS 16.0+
- A-series chip (A13+) or M-series chip
- ~3-5GB free storage
- Terminal app (a-Shell, iSH, or similar)

## Installation

### Using a-Shell (Recommended)

1. **Install a-Shell from App Store**
   - Download: [a-Shell on App Store](https://apps.apple.com/app/a-shell/id1473805438)

2. **Download and extract:**
```bash
curl -L https://github.com/akshaypn/qvac-finetune/releases/download/v1.0/qvac-ios-v1.0.zip -o qvac-ios.zip
unzip qvac-ios.zip
cd qvac-ios-v1.0
chmod +x bin/*
```

3. **Test installation:**
```bash
./bin/llama-cli --version
```

## Quick Start

### Download Model

For iOS devices, we recommend smaller models:

```bash
mkdir -p models
# Qwen3-0.6B (best for most devices)
curl -L https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/qwen3-0_6b-q8_0.gguf -o models/qwen3-0.6b-q8_0.gguf
```

### Run Inference

```bash
./bin/llama-cli \
  -m models/qwen3-0.6b-q8_0.gguf \
  -ngl 99 \
  -p "What is machine learning?"
```

### Fine-tune

```bash
# Download and extract biomedical dataset
curl -L https://github.com/tetherto/qvac-rnd-fabric-llm-finetune/raw/main/evaluation/biomedical_qa/biomedical_qa.zip -o biomedical_qa.zip
unzip biomedical_qa.zip

# Run LoRA fine-tuning
./bin/llama-finetune-lora \
  -m models/qwen3-0.6b-q8_0.gguf \
  -f biomedical_qa/train.jsonl \
  --assistant-loss-only \
  -c 128 -b 64 -ub 64 -ngl 99 -fa off \
  --lora-rank 8 --lora-alpha 16 \
  --num-epochs 3 \
  --checkpoint-save-steps 25
```

## Performance Expectations

### iPhone 15 Pro (A17 Pro)
- **Inference**: ~30-40 tokens/sec (0.6B Q8)
- **Training**: ~5-8 tokens/sec
- **Training Time**: ~1-2 hours for small datasets

### iPhone 14 Pro (A16)
- **Inference**: ~25-35 tokens/sec (0.6B Q8)
- **Training**: ~4-6 tokens/sec

### iPad Pro (M1/M2)
- **Inference**: Similar to macOS M-series
- **Training**: Faster than iPhone, similar to Mac

## Tips for iOS Training

### Battery Management
```bash
# Use lower batch sizes to reduce power consumption
-b 32 -ub 32

# Save checkpoints frequently in case of interruption
--checkpoint-save-steps 20

# Plug in device during training
```

### Memory Optimization
```bash
# Use smaller context
-c 64

# Lower LoRA rank
--lora-rank 4

# Use Q4_0 quantization for larger models
```

### Background Execution

⚠️ **Important**: iOS may suspend background tasks. Keep a-Shell in foreground during training.

```bash
# Enable checkpointing to avoid data loss
--checkpoint-save-steps 25 --checkpoint-save-dir "./checkpoints"
```

## Model Recommendations by Device

### iPhone 15 Pro / iPad Pro (M-series)
- ✅ Qwen3-0.6B Q8_0 (recommended)
- ✅ Qwen3-1.7B Q4_0 (may be slow)
- ⚠️ Qwen3-1.7B Q8_0 (high memory usage)

### iPhone 14 / iPhone 13
- ✅ Qwen3-0.6B Q8_0
- ✅ Qwen3-0.6B Q4_0 (faster)
- ❌ Models > 1B (will OOM)

### iPad Air / iPad mini
- ✅ Qwen3-0.6B Q8_0
- ⚠️ Qwen3-1.7B Q4_0 (depends on RAM)

## Dataset Format

### Instruction Fine-tuning
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain photosynthesis."},
    {"role": "assistant", "content": "Photosynthesis is..."}
  ]
}
```

## Advanced Configuration

### Full Training Command
```bash
./bin/llama-finetune-lora \
  -m models/qwen3-0.6b-q8_0.gguf \
  -f dataset.jsonl \
  --assistant-loss-only \
  -c 128 -b 64 -ub 64 -ngl 99 -fa off \
  --lora-rank 8 --lora-alpha 16 \
  --lora-modules "attn_q,attn_k,attn_v,attn_o" \
  --learning-rate 1e-5 --lr-min 1e-8 \
  --lr-scheduler cosine --warmup-ratio 0.1 \
  --checkpoint-save-steps 25 \
  --checkpoint-save-dir "./checkpoints" \
  --num-epochs 5
```

### Resume from Checkpoint
```bash
./bin/llama-finetune-lora \
  -m models/qwen3-0.6b-q8_0.gguf \
  -f dataset.jsonl \
  --resume-from "./checkpoints/checkpoint_step_00000075/" \
  --output-adapter final_adapter.gguf \
  -ngl 99
```

## Troubleshooting

### App Crashes or OOM
- Use smaller models (0.6B instead of 1.7B)
- Reduce context: `-c 32` or `-c 64`
- Use Q4_0 quantization
- Close all other apps
- Restart device and try again

### Slow Performance
- Ensure Metal backend is active (should be automatic)
- Check `-ngl 99` is set for GPU layers
- Reduce batch size if thermal throttling occurs
- Let device cool down between runs

### Permission Denied
```bash
chmod +x bin/*
```

### Storage Issues
```bash
# Clean up old checkpoints
rm -rf old_checkpoints/

# Use external storage if available
# (via Files app integration)
```

### Training Interrupted
```bash
# Resume from last checkpoint
--auto-resume

# Or specify checkpoint
--resume-from "./checkpoints/checkpoint_step_00000050/"
```

## Alternative Terminal Apps

### iSH
- More Linux-like environment
- Slower than a-Shell
- Better for debugging

### LibTerm
- Lightweight
- Good for simple tasks
- Limited package support

## Integration with iOS Apps

### Using Shortcuts
Create shortcuts to:
- Start training sessions
- Run inference with saved prompts
- Monitor training progress

### File Access
Models and datasets can be stored in:
- iCloud Drive
- On My iPhone/iPad
- Third-party cloud storage

## Performance Monitoring

### Check GPU Usage
```bash
# Monitor in real-time (if available)
./bin/llama-cli -m model.gguf -ngl 99 -p "test" --verbose
```

### Battery Impact
- Training can drain battery quickly
- Recommend plugging in device
- Use lower batch sizes for efficiency

## Use Cases

### Personal AI Assistant
Fine-tune on your writing style, preferences, and knowledge base.

### Domain-Specific Tools
Create specialized models for:
- Medical reference (biomedical dataset)
- Legal assistance
- Educational tutoring
- Language learning

### Privacy-Preserving Training
All data stays on your device - no cloud required.

## Limitations

- Background execution limited
- Battery consumption during training
- Storage constraints on some devices
- Cannot train models > 2B on most iPhones

## Support

- 📚 [Main Documentation](../../README.md)
- 📊 [Benchmarks](../../docs/BENCHMARKS.md)
- 🐛 [Report Issues](https://github.com/akshaypn/qvac-finetune/issues)
- 💬 [Discussions](https://github.com/akshaypn/qvac-finetune/discussions)
- 📱 [iOS-Specific Tips](https://github.com/akshaypn/qvac-finetune/wiki/iOS-Tips)

## License

MIT License - See [LICENSE](../../LICENSE) for details.

---

## Important Notes

1. **Keep device plugged in** during training
2. **Use checkpointing** to save progress frequently
3. **Test with small datasets** first to verify setup
4. **Monitor temperature** - let device cool if it gets hot
5. **Free up storage** - models and training data can be large

