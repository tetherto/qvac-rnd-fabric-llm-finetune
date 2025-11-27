# qvac-finetune Release Binaries

Pre-built binaries for all supported platforms - ready to use out of the box.


### 📦 What's Included

Each platform binary includes pre-built executables and libraries:
- ✅ `llama-finetune` - Full fine-tuning binary
- ✅ `llama-finetune-lora` - LoRA fine-tuning binary
- ✅ `llama-cli` - Inference and interactive chat
- ✅ `llama-quantize` - Model quantization tool
- ✅ `llama-perplexity` - Model evaluation tool
- ✅ `llama-export-lora` - Export/merge LoRA adapters
- ✅ All required shared libraries (GGML, backend libraries)

**Datasets & Examples:**
- 🧪 [Evaluation](../evaluation/) - Scripts, reports, datasets and examples
- 📖 [Documentation](../docs/) - Comprehensive benchmarks

---

## 🚀 Quick Download

| Platform | GPU/Hardware | Backend | Size | Download |
|----------|--------------|---------|------|----------|
| **Android** | Qualcomm Adreno, ARM Mali | Vulkan | 180MB | [📥 Download](./android/qvac-android-adreno-arm64-v1.0.zip) |
| **macOS** | Apple Silicon (M1-M4) | Metal | 35MB | [📥 Download](./macos/qvac-macos-apple-silicon-v1.0.zip) |
| **macOS** | Intel x64 | CPU | 36MB | [📥 Download](./macos/qvac-macos-intel-v1.0.zip) |
| **iOS** | Apple A-series | Metal | 1.3MB | [📥 Download](./ios/qvac-ios-v1.0.zip) |
| **Linux** | AMD/Intel/NVIDIA | Vulkan | 55MB | [📥 Download](./linux/qvac-linux-vulkan-x64-v1.0.zip) |
| **Linux** | ARM64 | CPU | 37MB | [📥 Download](./linux/qvac-linux-arm64-v1.0.zip) |
| **Linux** | Intel GPU | SYCL | 56MB | [📥 Download](./linux/qvac-linux-sycl-intel-v1.0.zip) |

---

## 📚 Platform-Specific Guides

Detailed installation and usage instructions for each platform:

- **[Android Guide](./android/README.md)** - Setup for Adreno and Mali GPUs
- **[macOS Guide](./macos/README.md)** - Apple Silicon and Intel Macs
- **[iOS Guide](./ios/README.md)** - iPhone and iPad setup
- **[Linux Guide](./linux/README.md)** - Vulkan, SYCL, and ARM64

---

## ✅ Platform Support Matrix

| Platform | Inference | Full Fine-tuning | LoRA Fine-tuning | Instruction Tuning | Checkpointing |
|----------|-----------|------------------|------------------|-------------------|---------------|
| Android (Adreno) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Android (Mali) | ✅ | ✅ | ✅ | ✅ | ✅ |
| macOS (M-series) | ✅ | ✅ | ✅ | ✅ | ✅ |
| macOS (Intel) | ✅ | ✅ | ✅ | ✅ | ✅ |
| iOS (A-series) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Linux (Vulkan) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Linux (SYCL) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Linux (ARM64) | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 🎯 Verified Model Support

All platforms have been tested with:

| Model Family | Sizes | Quantizations | Status |
|--------------|-------|---------------|--------|
| **Qwen3** | 0.6B, 1.7B, 4B | F32, F16, Q8_0, Q4_0 | ✅ Fully Supported |
| **Gemma-3** | 1B, 4B | F32, F16, Q8_0, Q4_0 | ✅ Fully Supported |
| **LLaMA 3.2** | 1B, 3B | F32, F16, Q8_0, Q4_0 | ✅ Fully Supported |
| **TinyLlama** | 1.1B | F32, F16, Q8_0, Q4_0 | ✅ Fully Supported |

---

## 📊 Performance Overview

### Inference Speed (tokens/second, Qwen3-1.7B Q8_0)

| Hardware | Peak | Average | TTFT |
|----------|------|---------|------|
| RTX 4090 | 180+ | 176 | 7ms |
| AMD 7900 XTX | 180 | 158 | 10ms |
| Apple M3 Pro | 90 | 62-90 | 37ms |
| Intel Arc A770 | 113 | 90 | 78ms |
| Adreno 830 | 35 | 17-24 | 609ms |
| Mali G715 | 8.3 | 7.8 | 795ms |

### Training Speed (tokens/second, Qwen3-1.7B Q8_0, LoRA)

| Hardware | Training t/s | Time/Epoch | Full Training (8 epochs) |
|----------|-------------|------------|--------------------------|
| RTX 4090 | 116 | 5.5 min | 45 min |
| AMD 7900 XTX | 47 | 13 min | 1.7 hrs |
| Apple M3 Pro | 17.5 | 40 min | 5.3 hrs |
| Intel Arc A770 | 30 | 20 min | 2.7 hrs |
| Adreno 830 | 6 | 1h 40min | 13 hrs |
| Mali G715 | 1.3 | 7h 40min | 61 hrs |

> 📊 See [complete benchmarks](../docs/BENCHMARKS.md) for detailed metrics

---

## 🔬 Quality Validation

### vs PyTorch/HuggingFace

| Metric | qvac-finetune | PyTorch |
|--------|---------------|---------|
| **LLM-as-Judge Win Rate** | 45-48% | 52-55% |
| **Biomedical Accuracy** | 79-94% | 78-86% |
| **Cosine Similarity** | 0.82 | 0.77 |
| **Jaccard Similarity** | 0.19 | 0.23 |

**Conclusion:** Near-parity with established frameworks, proving cross-platform training maintains quality.

---

## 🆕 What's New in v1.0

### Core Features
- ✅ **Cross-Platform LoRA Training** - Works on all modern GPUs
- ✅ **Instruction Fine-Tuning** - Masked-loss training for alignment
- ✅ **Dynamic Tiling** - Solves Adreno GPU memory constraints
- ✅ **Checkpointing System** - Save and resume training
- ✅ **Learning Rate Scheduling** - Cosine, linear, and constant schedulers

### Architecture Support
- ✅ **GEGLU Backward Pass** - Enables Gemma fine-tuning
- ✅ **OUT_PROD Operator** - Full GPU support (CUDA, Vulkan, Metal)
- ✅ **Quantized Training** - Q4_0 and Q8_0 fine-tuning
- ✅ **Mixed Precision** - FP16 and FP32 training

### Platform Enhancements
- ✅ **Metal Backend** - Native Apple GPU support (iOS, macOS)
- ✅ **Vulkan Enhancements** - AMD, Intel, NVIDIA, Adreno, Mali
- ✅ **SYCL Support** - Optimized Intel GPU backend
- ✅ **ARM64** - Raspberry Pi and ARM server support

### Data Pipeline
- ✅ **ChatML Templates** - Built-in conversation formatting
- ✅ **Custom Jinja** - Flexible template system
- ✅ **JSONL Support** - HuggingFace-compatible datasets
- ✅ **Masked Loss** - Train only on assistant responses

---

## 📖 Documentation

### Getting Started
- [Quick Start Guide](../README.md#quick-start)
- [Installation Instructions](../README.md#installation)
- [First Fine-Tuning Session](../README.md#finetune-lora)

### Platform Guides
- [Android Setup & Tips](./android/README.md)
- [macOS Setup & Tips](./macos/README.md)
- [iOS Setup & Tips](./ios/README.md)
- [Linux Setup & Tips](./linux/README.md)

### Advanced Topics
- [Detailed Benchmarks](../docs/BENCHMARKS.md)
- [Research Paper](../docs/paper.pdf)
- [API Reference](../docs/API.md)
- [Dataset Format](../datasets/README.md)

---

## 🐛 Known Issues

### Mobile Platforms
- ⚠️ Qwen3-4B causes OOM on most mobile devices (use 1.7B or smaller)
- ⚠️ iOS may suspend background training (keep app in foreground)
- ⚠️ Mali G715 training is slower than Adreno (functional but patience required)

### Desktop Platforms
- ⚠️ Flash attention not yet supported on Vulkan backend
- ⚠️ Multi-GPU training experimental (single GPU recommended)

### General
- ⚠️ Very large batch sizes (>256) may cause OOM on some GPUs
- ⚠️ WebGPU backend is experimental

### Workarounds
Most issues can be resolved by:
- Using smaller models or context windows
- Reducing batch size
- Using Q4_0 quantization
- Enabling checkpointing for long runs

---

## 🔄 Upgrade Path

### From Previous Versions
This is the initial v1.0 release. Future versions will maintain backward compatibility.

### Model Compatibility
All models fine-tuned with v1.0 are compatible with:
- Future qvac-finetune versions
- llama.cpp inference
- Any GGUF-compatible runtime

---

## 💡 Quick Start Examples

### Android (Termux)
```bash
export LD_LIBRARY_PATH=.
./bin/llama-finetune-lora \
  -m qwen3-0.6b-q8_0.gguf \
  -f biomedical_qa.jsonl \
  --assistant-loss-only \
  -c 128 -b 64 -ub 64 -ngl 99 -fa off \
  --num-epochs 2
```

### macOS (Apple Silicon)
```bash
./bin/llama-finetune-lora \
  -m qwen3-1.7b-q8_0.gguf \
  -f biomedical_qa.jsonl \
  --assistant-loss-only \
  -c 128 -b 128 -ub 128 -ngl 999 -fa off \
  --num-epochs 8 \
  --lora-modules "attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down"
```

### Linux (Vulkan)
```bash
./bin/llama-finetune-lora \
  -m qwen3-1.7b-q8_0.gguf \
  -f biomedical_qa.jsonl \
  --assistant-loss-only \
  -c 128 -b 128 -ub 128 -ngl 999 -fa off \
  --learning-rate 1e-5 --lr-scheduler cosine \
  --checkpoint-save-steps 50 \
  --num-epochs 8
```

---

## 🙏 Acknowledgments

Built on the llama.cpp foundation with extensive contributions:
- GGML core engine enhancements
- Vulkan backend improvements
- Metal backend training support
- Dynamic tiling algorithm for mobile GPUs
- LoRA architecture integration

Special thanks to the llama.cpp community and all hardware vendors who provided testing devices.

---

## 📞 Support & Community

- 🌐 [Project Website](https://github.com/akshaypn/qvac-finetune)
- 💬 [GitHub Discussions](https://github.com/akshaypn/qvac-finetune/discussions)
- 🐛 [Issue Tracker](https://github.com/akshaypn/qvac-finetune/issues)
- 📖 [Documentation](../README.md)
- 📊 [Benchmarks](../docs/BENCHMARKS.md)
- 🔬 [Research Paper](../docs/paper.pdf)

---

## 📄 License

MIT License - See [LICENSE](../LICENSE) for details.

**Dataset Licenses:**
- Biomedical QA: MIT (PubMedQA-derived)
- Email Dataset: Internal research use

---

## 🔮 Roadmap

### Upcoming Features
- [ ] Multi-GPU training support
- [ ] Flash attention on Vulkan
- [ ] WebGPU backend stabilization
- [ ] Additional model architectures
- [ ] Distributed training
- [ ] Quantization-aware training improvements

### Community Requests
Vote on features in [GitHub Discussions](https://github.com/akshaypn/qvac-finetune/discussions)!

---

<div align="center">
  <p><b>Making LLM fine-tuning accessible to everyone, everywhere</b></p>
  <p>From smartphones to datacenters • No vendor lock-in • Privacy-preserving</p>
  <p>⭐ Star the repo if you find it useful!</p>
</div>

