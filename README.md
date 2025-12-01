<p align="center">
  <img src="https://img.shields.io/badge/Framework-llama.cpp%2FGGML-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/LoRA%20Fine--Tuning-Supported-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Hardware-Mobile%20%7C%20Desktop%20%7C%20Server-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Backends-Vulkan%20%7C%20Metal%20%7C%20CUDA-purple?style=for-the-badge"/>
</p>

<h1 align="center">An Edge-First Generalized LLM LoRA Fine-Tuning Framework for Heterogeneous GPUs</h1>

<div align="center">
  <b>The first truly cross-platform LoRA fine-tuning solution for Large Language Models</b><br>
  <b>From smartphones to datacenters • No vendor lock-in • Privacy-preserving on-device training</b>
</div>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-downloads">Downloads</a> •
  <a href="#-datasets">Datasets</a> •
  <a href="#-research-highlights">Research</a> •
  <a href="#-performance-benchmarks">Benchmarks</a>
</p>

---

## 🎯 What Makes This Different?

**The Problem:** LLM fine-tuning has been locked to NVIDIA GPUs and CUDA. Mobile devices, AMD/Intel GPUs, and Apple Silicon were left behind.

**Our Solution:** A unified LoRA fine-tuning framework that works on **any** modern GPU:

| Platform | Hardware |
|----------|----------|
| 📱 **Android** | Qualcomm Adreno, ARM Mali |
| 🍎 **iOS/macOS** | Apple Silicon (A-series, M-series) |
| 🖥️ **Windows/Linux** | AMD, Intel, NVIDIA GPUs |

**Key Innovation:** Novel dynamic tiling algorithm enables stable training on mobile GPUs with hardware memory constraints.

---

## 🔬 Research Highlights

This repository contains the implementation and artifacts for our paper:

**["An Edge-First Generalized LLM LoRA Fine-Tuning Framework for Heterogeneous GPUs"](https://huggingface.co/blog/qvac/fabric-llm-finetune)**

### Key Contributions

1. **🌍 Cross-Platform LoRA Framework** - First unified solution for parameter-efficient fine-tuning across heterogeneous consumer hardware
2. **📱 Mobile GPU Support** - First successful fine-tuning on Adreno, Mali, and Apple mobile GPUs
3. **🎓 Instruction-Tuning** - Masked-loss training for instruction-following alignment
4. **⚡ Modern Architecture Support** - Extended llama.cpp to support Qwen3 and Gemma3 fine-tuning
5. **🔧 Hardware Innovation** - Dynamic tiling algorithm solves critical Adreno GPU memory constraints

## 🚀 Empowering the Community with Open Resources

To accelerate development and innovation, **Tether Data** is publicly releasing:

- **Fine‑tuned Model Adapters**  
  👉 [fabric‑llm‑finetune on Hugging Face](https://huggingface.co/qvac/fabric-llm-finetune)

- **Source Code (Work‑in‑Progress)**  
  👉 [qvac‑fabric‑llm.cpp (fabric‑llm‑finetune branch)](https://github.com/tetherto/qvac-fabric-llm.cpp/tree/fabric-llm-finetune)  
  *Currently experimental and intended for developers to extend the solution for other LLM models.*

### Validated Performance

- ✅ **Quality Parity**: 45-48% win rate vs PyTorch/HuggingFace (LLM-as-judge)
- ✅ **Domain Adaptation**: 79-94% accuracy on biomedical Q&A tasks
- ✅ **Production Scale**: Tested on 6 GPU architectures, 5 model families, 4 quantization levels

> 📊 [View detailed benchmarks](./docs/BENCHMARKS.md) | 📄 Research paper: Coming soon

---
### 🗺️ Navigation Guide: Where to Find What

#### 🚀 **Getting Started**
- **First time?** Start with [Quick Start](#quick-start) section above
- **Platform-specific setup?** Go to `releases/[your-platform]/README.md`
- **Download binaries?** Browse `releases/` directory

#### 📊 **Datasets & Examples**
- **Training datasets:** `evaluation/email_style_transfer/`
- **Dataset format guide:** `evaluation/email_style_transfer/README.md`
- **How to perform custom finetuning:** `evaluation/README.md`

#### 🧪 **Evaluation & Testing**
- **Run model comparisons:** Use scripts in `evaluation/scripts/`
- **View benchmark results:** `docs/BENCHMARKS.md` (comprehensive)
- **Detailed experiment reports:** `evaluation/reports/` directory
- **Compare base vs fine-tuned:** `evaluation/scripts/compare_base_vs_adapters.py`

#### 📖 **Documentation & Research**
- **Complete benchmarks:** `docs/BENCHMARKS.md` (all platforms, metrics)
- **Methodology & results:** `evaluation/reports/COMPLETE_PROJECT_REPORT.md`
- **Biomedical case study:** `evaluation/reports/BIOMED_FINETUNING_REPORT.md`

#### 💡 **Common Tasks**
| Task | Location |
|------|----------|
| Download binaries | `releases/[platform]/` |
| Get training data | `evaluation/email_style_transfer/email_dataset.jsonl` |
| See platform benchmarks | `docs/BENCHMARKS.md` |
| Run evaluation scripts | `evaluation/scripts/` |
| View experiment results | `evaluation/reports/` |
| Platform setup guide | `releases/[platform]/README.md` |

## 📁 Repository Structure

```
qvac-fabric/
├── README.md                      # This file - main documentation
├── RELEASE_NOTES.md               # Version history and changelog
│
├── docs/                          # 📖 Research Documentation
│   └── BENCHMARKS.md              # Comprehensive performance metrics across all platforms
│
├── evaluation/                    # 🧪 Datasets, Scripts & Results
│   ├── README.md                  # Evaluation guide and methodology
│   │
│   ├── email_style_transfer/      # Personal Email Style Transfer Dataset
│   │   ├── email_dataset.jsonl    # Email conversation examples
│   │   └── README.md              # Usage and format documentation
│   │
│   ├── scripts/                   # Python Evaluation & Monitoring Tools
│   │   ├── compare_base_vs_adapters.py    # Compare base model vs fine-tuned
│   │   ├── monitor_training.py            # Real-time training monitoring
│   │   ├── quick_compare.py               # Quick model comparison
│   │   ├── test_biomed_prompts.py         # Test prompts accuracy
│   │   └── build_biomed_yn_dataset.py     # Dataset preprocessing
│   │
│   └── reports/                   # Experimental Results & Analysis
│       ├── BIOMED_FINETUNING_REPORT.md        # Detailed results
│       ├── COMPLETE_PROJECT_REPORT.md         # Full project overview
│       ├── README_inference_comparison.md     # Inference benchmarks
│       └── biomed_comparison_results.json     # Structured results data
│
└── releases/                      # 📦 Pre-built Binaries
    ├── README.md                  # Platform overview and installation
    │
    ├── android/                   # Android (Termux) Builds
    │   ├── README.md              # Android-specific setup guide
    │   └── qvac-android-adreno-arm64-v1.0.zip
    │
    ├── ios/                       # iOS Builds
    │   ├── README.md              # iOS-specific setup guide
    │   └── qvac-ios-v1.0.zip
    │
    ├── linux/                     # Linux Builds (Multiple Backends)
    │   ├── README.md              # Linux setup and backend selection
    │   ├── qvac-linux-arm64-v1.0.zip         # ARM64 CPU build
    │   ├── qvac-linux-vulkan-x64-v1.0.zip    # AMD/NVIDIA/Intel Vulkan
    │   └── qvac-linux-sycl-intel-v1.0.zip    # Intel GPU SYCL
    │
    └── macos/                     # macOS Builds
        ├── README.md              # macOS setup guide
        ├── qvac-macos-apple-silicon-v1.0.zip  # M1/M2/M3/M4
        └── qvac-macos-intel-v1.0.zip          # Intel x64
```



## 🚀 Quick Start

### Choose Your Platform

<details>
<summary><b>📱 Android (Termux)</b></summary>

```bash
# Download pre-built binary for your device
wget https://github.com/tetherto/qvac-fabric/releases/download/v1.0/qvac-android-adreno-arm64-v1.0.zip
unzip qvac-android-adreno-arm64-v1.0.zip
cd qvac-android-adreno-arm64-v1.0

# Set library path (required for Android)
export LD_LIBRARY_PATH=.

# Download model
mkdir -p models
wget https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/qwen3-0_6b-q8_0.gguf -O models/qwen3-0.6b-q8_0.gguf

# Download dataset
wget https://raw.githubusercontent.com/tetherto/qvac-fabric/main/datasets/train.jsonl

# Quick test with biomedical dataset
./bin/llama-finetune-lora \
  -m models/qwen3-0.6b-q8_0.gguf \
  -f train.jsonl \
  --assistant-loss-only \
  -c 128 -b 128 -ub 128 -ngl 99 -fa off \
  --num-epochs 2
```

**📖 [Full Android Guide](./releases/android/README.md)**

</details>

<details>
<summary><b>🍎 macOS (Apple Silicon)</b></summary>

```bash
# Download pre-built binary
curl -L https://github.com/tetherto/qvac-fabric/releases/download/v1.0/qvac-macos-apple-silicon-v1.0.zip -o qvac-macos.zip
unzip qvac-macos.zip
cd qvac-macos-apple-silicon-v1.0

# Download model
mkdir -p models
wget https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/qwen3-1_7b-q8_0.gguf -O models/qwen3-1.7b-q8_0.gguf

# Download dataset
wget https://raw.githubusercontent.com/tetherto/qvac-fabric/main/datasets/train.jsonl

# Quick test with email style transfer
./bin/llama-finetune-lora \
  -m models/qwen3-1.7b-q8_0.gguf \
  -f train.jsonl \
  -c 512 -b 128 -ub 128 -ngl 999 \
  --lora-rank 16 --lora-alpha 32 \
  --num-epochs 3
```

**📖 [Full macOS Guide](./releases/macos/README.md)**

</details>

<details>
<summary><b>🖥️ Linux/Windows (AMD/Intel/NVIDIA)</b></summary>

```bash
# Download binary for your GPU
# For AMD/Intel/NVIDIA (Vulkan):
wget https://github.com/tetherto/qvac-fabric/releases/download/v1.0/qvac-linux-vulkan-x64-v1.0.zip

unzip qvac-linux-vulkan-x64-v1.0.zip
cd qvac-linux-vulkan-x64-v1.0

# Download model
mkdir -p models
wget https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/qwen3-1_7b-q8_0.gguf -O models/qwen3-1.7b-q8_0.gguf

# Download dataset
wget https://raw.githubusercontent.com/tetherto/qvac-fabric/main/datasets/train.jsonl

# Run biomedical fine-tuning
./bin/llama-finetune-lora \
  -m models/qwen3-1.7b-q8_0.gguf \
  -f train.jsonl \
  --assistant-loss-only \
  -c 128 -b 128 -ub 128 -ngl 999 -fa off \
  --learning-rate 1e-5 --lr-min 1e-8 \
  --lr-scheduler cosine --warmup-ratio 0.1 \
  --num-epochs 8 \
  --lora-modules "attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down"
```

**📖 [Full Linux Guide](./releases/linux/README.md)**

</details>

---

## 📦 Downloads

Pre-built binaries optimized for each platform:

| Platform | Hardware | Backend | Size | Download |
|----------|----------|---------|------|----------|
| Android | Qualcomm Adreno, ARM Mali | Vulkan | 180MB | [📥 Download](./releases/android/qvac-android-adreno-arm64-v1.0.zip) |
| macOS | Apple M1/M2/M3/M4 | Metal | 35MB | [📥 Download](./releases/macos/qvac-macos-apple-silicon-v1.0.zip) |
| macOS | Intel x64 | CPU | 36MB | [📥 Download](./releases/macos/qvac-macos-intel-v1.0.zip) |
| iOS | Apple A-series | Metal | 1.3MB | [📥 Download](./releases/ios/qvac-ios-v1.0.zip) |
| Linux/Win | AMD/Intel/NVIDIA | Vulkan | 55MB | [📥 Download](./releases/linux/qvac-linux-vulkan-x64-v1.0.zip) |
| Linux | ARM64 | CPU | 37MB | [📥 Download](./releases/linux/qvac-linux-arm64-v1.0.zip) |
| Linux | Intel GPU | SYCL | 56MB | [📥 Download](./releases/linux/qvac-linux-sycl-intel-v1.0.zip) |

### What's Included

Each download contains pre-built binaries:
- ✅ `llama-finetune-lora` - LoRA fine-tuning binary
- ✅ `llama-finetune` - Full fine-tuning binary
- ✅ `llama-cli` - Inference and interactive chat
- ✅ `llama-quantize` - Model quantization tool
- ✅ `llama-perplexity` - Model evaluation tool
- ✅ `llama-export-lora` - Export/merge LoRA adapters
- ✅ All required libraries (GGML, Vulkan/Metal backends)

**Note:** Datasets and examples are available in the [evaluation](./evaluation/) directory of this repository.

**📖 [All Releases & Documentation](./releases/README.md)**

---

## 📊 Datasets

We provide curated, privacy-safe datasets for reproducible fine-tuning research. See the `evaluation/` directory for available datasets and documentation.

---

## 🎯 Key Features

### Training Capabilities

- 🎯 **Full Fine-tuning & LoRA** - Support for both full model updates and parameter-efficient LoRA
- 🔄 **Instruction Fine-Tuning** - Masked-loss training on assistant tokens only
- 📝 **Chat Templates** - Built-in ChatML + custom Jinja template support
- 💾 **Checkpointing** - Resume training with complete optimizer state
- 📊 **Learning Rate Scheduling** - Cosine annealing with warmup
- 📦 **Quantization** - Train and infer with F32, F16, Q8_0, Q4_0

### Architecture Support

- ✅ Qwen3 (0.6B, 1.7B, 4B)
- ✅ Gemma-3 (1B, 4B)
- ✅ LLaMA family
- ✅ TinyLlama

### Hardware Backends

- 🔷 **Vulkan** - AMD, Intel, NVIDIA, Qualcomm Adreno, ARM Mali
- 🍎 **Metal** - Apple Silicon (M-series, A-series)
- 💚 **CUDA** - NVIDIA GPUs (optional, Vulkan works too)
- 🖥️ **CPU** - Fallback for any platform

---

## 📈 Performance Benchmarks

### Inference Speed (tokens/second)

| Model | Mali | Adreno | Intel A770 | AMD 7900XTX | RTX 4090 | Apple M3 | iPhone 16 |
|-------|------|--------|------------|-------------|----------|----------|-----------|
| **Qwen3-0.6B Q8** | 15.4 | 35.0 | 133.0 | 178.2 | 199+ | 120+ | 32.5 |
| **Qwen3-1.7B Q8** | 7.8 | 17.3 | 90.0 | 158.0 | 176+ | 62-90 | 15.2 |
| **Gemma-1B Q8** | 11.7 | 36.6 | 89.0 | 148.8 | 150+ | 70-90 | 33.2 |

### Fine-tuning Speed (Time per Epoch, Qwen3-1.7B Q8)

| Hardware | Time/Epoch | Full Training (8 epochs) |
|----------|------------|--------------------------|
| **RTX 4090** | 5.5 min | **45 min** ⚡ |
| **AMD 7900 XTX** | 13 min | 1.7 hrs |
| **Intel Arc A770** | 20 min | 2.7 hrs |
| **Apple M3 Pro** | 40 min | 5.3 hrs |
| **iPhone 16** | 1h 55min | 15 hrs |
| **Adreno 830** | 1h 40min | 13 hrs |
| **Mali G715** | 7h 40min | 61 hrs |

> 📊 [View complete benchmarks](./docs/BENCHMARKS.md) with detailed metrics across all platforms

### Quality Comparison vs PyTorch

| Metric | qvac-fabric | PyTorch/HuggingFace |
|--------|---------------|---------------------|
| **LLM-as-Judge Win Rate** | 45-48% | 52-55% |
| **Biomedical Accuracy** | 79-94% | 78-86% |
| **Cosine Similarity** | 0.82 | 0.77 |

**Conclusion:** Near-parity quality with established frameworks, but works on **8x more hardware platforms**.

---

## 🔧 Usage Examples

### Basic LoRA Fine-Tuning

```bash
# Create new LoRA adapter
./bin/llama-finetune-lora \
  -m model.gguf \
  -f dataset.txt \
  -ngl 999 -c 512 -b 512 -ub 512 -fa off
```

### Custom LoRA Configuration

```bash
# Advanced LoRA parameters
./bin/llama-finetune-lora \
  -m model.gguf \
  -f dataset.txt \
  -ngl 999 -c 512 -b 512 -ub 512 \
  --lora-rank 16 --lora-alpha 32 \
  --lora-modules "attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down" \
  -fa off
```

### Instruction Fine-Tuning (SFT)

```bash
# Train only on assistant responses
./bin/llama-finetune-lora \
  -m model.gguf \
  -f conversations.jsonl \
  --assistant-loss-only \
  --chat-template custom.jinja \
  -ngl 999 -c 512 -b 128 -ub 128 -fa off
```

### Checkpointing & Resume

```bash
# Save checkpoints every 50 steps
./bin/llama-finetune-lora \
  -m model.gguf \
  -f dataset.txt \
  --checkpoint-save-steps 50 \
  --checkpoint-save-dir "./checkpoints" \
  -ngl 999

# Resume from checkpoint
./bin/llama-finetune-lora \
  -m model.gguf \
  -f dataset.txt \
  --resume-from "./checkpoints/checkpoint_step_00000150/" \
  --output-adapter improved_adapter.gguf \
  -ngl 999
```

### Using Trained Adapters

```bash
# Inference with LoRA adapter
./bin/llama-cli \
  -m base_model.gguf \
  --lora trained_adapter.gguf \
  -ngl 999 \
  -p "Your prompt here"
```

---

## 🏗️ Technical Architecture

### LoRA Integration

Our implementation augments pretrained weights with low-rank updates:

```
W' = W + α(AB)
```

Where:
- **W**: Frozen base model weights
- **A** ∈ ℝ^(d×r), **B** ∈ ℝ^(r×d): Trainable low-rank matrices
- **r**: LoRA rank (typically 8-32)
- **α**: Scaling factor

Only matrices **A** and **B** are updated during training, reducing parameters by orders of magnitude.

### Dynamic Tiling Algorithm

**Problem:** Adreno GPUs have undocumented 128MiB SSBO limit causing `DeviceLoss` errors.

**Solution:** Dynamically tile large matrix operations based on input shapes:

1. Calculate tile dimensions that respect 128MiB limit
2. Execute operations on tiles independently
3. Assemble results into final output tensor

This enables stable training on mobile GPUs where static approaches fail.

### Backend Architecture

```
┌─────────────────────────────────────┐
│    llama.cpp Public API             │
│  (llama_lora_training_init, etc.)   │
├─────────────────────────────────────┤
│         GGML Core Engine            │
│  (Forward/Backward Pass, Optimizer) │
├──────────┬──────────┬───────────────┤
│  Vulkan  │  Metal   │    CUDA       │
│ (Cross)  │ (Apple)  │  (NVIDIA)     │
└──────────┴──────────┴───────────────┘
     ↓           ↓            ↓
┌──────────┬──────────┬───────────────┐
│  Adreno  │  Apple   │    RTX        │
│   Mali   │  M/A     │    AMD        │
│  Intel   │  Series  │    etc.       │
└──────────┴──────────┴───────────────┘
```

---

## 📚 Documentation

### Getting Started
- [Installation Guide](#-downloads)
- [Quick Start](#-quick-start)
- [Platform-Specific Guides](./releases/)

### Advanced Topics
- [Detailed Benchmarks](./docs/BENCHMARKS.md)
- [Email Dataset Documentation](./evaluation/email_style_transfer/README.md)
- [Evaluation Guide](./evaluation/README.md)

### Platform Guides
- [Android Setup](./releases/android/README.md)
- [macOS Setup](./releases/macos/README.md)
- [iOS Setup](./releases/ios/README.md)
- [Linux Setup](./releases/linux/README.md)

---

## 🤝 Contributing

We welcome contributions! Areas of interest:

- 🔧 Optimizations for specific GPU architectures
- 📱 Testing on additional mobile devices
- 🏗️ Support for new model architectures
- 📊 Benchmark contributions
- 📝 Documentation improvements

Please open an issue or pull request on GitHub to contribute.

---

## 🐛 Known Issues

### Mobile Platforms
- ⚠️ Qwen3-4B causes OOM on most mobile devices → Use 1.7B or smaller
- ⚠️ iOS may suspend background training → Keep app in foreground
- ⚠️ Mali G715 training slower than Adreno → Functional but requires patience

### Desktop Platforms
- ⚠️ Flash attention not yet supported on Vulkan → Use `-fa off`
- ⚠️ Multi-GPU training experimental → Use single GPU

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{qvac-fabric,
  title={An Edge-First Generalized LLM LoRA Fine-Tuning Framework for Heterogeneous GPUs},
  author={[Subash, Akshay, Patrik, Milan, Nurman]},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 🙏 Acknowledgments

This work builds on:
- **llama.cpp** - Foundation inference engine
- **LoRA** (Hu et al., 2021) - Parameter-efficient fine-tuning method
- **PubMedQA** - Jin, Qiao, et al. "Pubmedqa: A dataset for biomedical research question answering." *Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP)*. 2019.


---

## 📄 License

This project is licensed under the Apache 2.0 License.

---

## 🔗 Links

- 🌐 [Project Website](https://github.com/tetherto/qvac-fabric)
- 📦 [Release Downloads](./releases)
- 💬 [Discussion Forum](https://github.com/tetherto/qvac-fabric/discussions)
- 🐛 [Issue Tracker](https://github.com/tetherto/qvac-fabric/issues)
- 📖 [Full Documentation](./docs/)

---

<div align="center">
  <h2>Making LLM fine-tuning accessible to everyone, everywhere</h2>
  <p><b>From smartphones to datacenters • No vendor lock-in • Privacy-preserving</b></p>
  <p>⭐ Star this repo if you find it useful!</p>
</div>
