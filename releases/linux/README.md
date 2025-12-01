# Linux Binaries

## Vulkan Backend (Universal)

Works with AMD, Intel, and NVIDIA GPUs through the Vulkan API.

### Tested GPUs
- **AMD**: Radeon RX 7900 XTX
- **Intel**: Arc A770
- **NVIDIA**: RTX 4090

### Requirements
- Ubuntu 22.04+ or equivalent
- Vulkan drivers installed
- ~2-4GB free storage

### Installation

```bash
# Download binary
wget https://github.com/akshaypn/qvac-finetune/releases/download/v1.0/qvac-linux-vulkan-x64-v1.0.zip
unzip qvac-linux-vulkan-x64-v1.0.zip
cd qvac-linux-vulkan-x64-v1.0
chmod +x bin/*
```

### Verify Vulkan

```bash
# Check Vulkan installation
vulkaninfo | grep "GPU"

# If vulkaninfo not found, install:
# Ubuntu/Debian:
sudo apt install vulkan-tools

# Fedora:
sudo dnf install vulkan-tools

# Arch:
sudo pacman -S vulkan-tools
```

### Quick Start

```bash
# Download model
mkdir -p models
wget https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/qwen3-1_7b-q8_0.gguf -O models/qwen3-1.7b-q8_0.gguf

# Test inference
./bin/llama-cli -m models/qwen3-1.7b-q8_0.gguf -ngl 999 -p "Hello, world!"

# Fine-tune
wget https://raw.githubusercontent.com/akshaypn/qvac-finetune/main/datasets/biomedical_qa.jsonl

./bin/llama-finetune-lora \
  -m models/qwen3-1.7b-q8_0.gguf \
  -f biomedical_qa.jsonl \
  --assistant-loss-only \
  -c 128 -b 128 -ub 128 -ngl 999 -fa off \
  --learning-rate 1e-5 --lr-min 1e-8 \
  --lr-scheduler cosine --warmup-ratio 0.1 \
  --num-epochs 8 \
  --lora-modules "attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down"
```

### Performance by GPU

#### AMD Radeon RX 7900 XTX
- **Qwen3-1.7B Q8**: 158 tokens/sec inference, 47 tokens/sec training
- **Training Time**: ~13 minutes per epoch
- **Full 8 epochs**: ~1.7 hours

#### Intel Arc A770
- **Qwen3-1.7B Q8**: 90 tokens/sec inference, 30 tokens/sec training
- **Training Time**: ~20 minutes per epoch
- **Full 8 epochs**: ~2.7 hours

#### NVIDIA RTX 4090
- **Qwen3-1.7B Q8**: 180+ tokens/sec inference, 116 tokens/sec training
- **Training Time**: ~5-6 minutes per epoch
- **Full 8 epochs**: ~45 minutes

---

## Intel SYCL Backend

Optimized for Intel GPUs using oneAPI SYCL.

### Requirements
- Intel GPU (Arc series or newer)
- Intel oneAPI toolkit
- Ubuntu 22.04+

### Install Intel oneAPI

```bash
# Add Intel repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

# Install oneAPI
sudo apt update
sudo apt install intel-oneapi-compiler-dpcpp-cpp
```

### Installation

```bash
wget https://github.com/akshaypn/qvac-finetune/releases/download/v1.0/qvac-linux-sycl-intel-v1.0.zip
unzip qvac-linux-sycl-intel-v1.0.zip
cd qvac-linux-sycl-intel-v1.0
chmod +x bin/*

# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh
```

### Performance
- Optimized specifically for Intel GPUs
- Better performance than Vulkan on Intel Arc
- Direct hardware access for maximum efficiency

---

## ARM64 (CPU)

For ARM-based Linux systems (Raspberry Pi 5, etc.).

### Requirements
- ARM64 Linux (Ubuntu/Debian)
- 4GB+ RAM (8GB+ recommended)

### Installation

```bash
wget https://github.com/akshaypn/qvac-finetune/releases/download/v1.0/qvac-linux-arm64-v1.0.zip
unzip qvac-linux-arm64-v1.0.zip
cd qvac-linux-arm64-v1.0
chmod +x bin/*
```

### Performance Note
CPU-only performance is slower. Recommended for smaller models only.

```bash
# Use Q4_0 models for better CPU performance
wget https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/qwen3-0_6b-q4_0.gguf

# Inference (no -ngl flag for CPU)
./bin/llama-cli -m qwen3-0_6b-q4_0.gguf -p "Hello"

# Training with minimal settings
./bin/llama-finetune-lora \
  -m qwen3-0_6b-q4_0.gguf \
  -f dataset.jsonl \
  -c 64 -b 32 -ub 32 \
  --lora-rank 4 \
  --num-epochs 2
```

---

## Common Configuration

### Full LoRA Fine-tuning

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
  --output-adapter trained_adapter.gguf \
  --num-epochs 8
```

### Checkpointing

```bash
# Save checkpoints every 50 steps
--checkpoint-save-steps 50 --checkpoint-save-dir "./checkpoints"

# Resume training
./bin/llama-finetune-lora \
  -m models/model.gguf \
  -f dataset.jsonl \
  --resume-from "./checkpoints/checkpoint_step_00000150/" \
  --output-adapter improved_adapter.gguf \
  -ngl 999
```

### Using Trained Adapters

```bash
# Inference with adapter
./bin/llama-cli \
  -m models/base_model.gguf \
  --lora trained_adapter.gguf \
  -ngl 999 \
  -p "Your prompt here"
```

## Troubleshooting

### Vulkan Not Found

```bash
# Ubuntu/Debian
sudo apt install vulkan-tools mesa-vulkan-drivers

# For AMD
sudo apt install mesa-vulkan-drivers

# For Intel
sudo apt install intel-media-va-driver-non-free

# For NVIDIA
# Vulkan is included in nvidia-driver package
sudo apt install nvidia-driver-535
```

### GPU Not Detected

```bash
# Check Vulkan devices
vulkaninfo | grep "deviceName"

# Check GPU is visible
lspci | grep -i vga

# Verify drivers loaded
lsmod | grep -i nvidia  # For NVIDIA
lsmod | grep -i amdgpu  # For AMD
lsmod | grep -i i915    # For Intel
```

### Out of Memory

```bash
# Reduce context
-c 256

# Lower batch size
-b 64 -ub 64

# Use Q4_0 quantization
# Download Q4_0 model instead of Q8_0

# Reduce LoRA rank
--lora-rank 8
```

### Slow Performance

```bash
# Ensure GPU offload is enabled
-ngl 999

# Check GPU usage
nvidia-smi  # For NVIDIA
radeontop   # For AMD
intel_gpu_top  # For Intel

# Monitor in real-time
watch -n 1 nvidia-smi
```

### Permission Errors

```bash
chmod +x bin/*

# If still issues, check SELinux
sudo setenforce 0  # Temporarily disable
```

## Model Recommendations

### High-end GPUs (RTX 4090, RX 7900 XTX)
- ✅ Qwen3-4B Q8_0
- ✅ Gemma-4B Q8_0
- ✅ All smaller models

### Mid-range GPUs (RTX 3060, RX 6700, Arc A770)
- ✅ Qwen3-1.7B Q8_0 (recommended)
- ✅ Gemma-1B Q8_0
- ⚠️ Qwen3-4B Q4_0 (may be slow)

### Entry GPUs / CPU
- ✅ Qwen3-0.6B Q4_0
- ✅ Qwen3-1.7B Q4_0
- ⚠️ Models > 2B (very slow)

## Dataset Format

### Instruction Fine-tuning (JSONL)
```json
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is..."}]}
```

### Plain Text
```
Line 1 of training data
Line 2 of training data
...
```

## Advanced Features

### Multi-GPU Training (Experimental)

```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1
# Or for Vulkan
export VK_DEVICE_ID=0
```

### Custom Learning Rate Schedule

```bash
# Cosine with warmup (recommended)
--lr-scheduler cosine --warmup-ratio 0.1

# Linear decay
--lr-scheduler linear --warmup-ratio 0.05

# Constant learning rate
--lr-scheduler constant
```

### Export Merged Model

```bash
./bin/llama-export-lora \
  -m base_model.gguf \
  --lora adapter.gguf \
  -o merged_model.gguf
```

## Performance Monitoring

### NVIDIA
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Detailed metrics
nvidia-smi dmon -s pucvmet
```

### AMD
```bash
# Install radeontop
sudo apt install radeontop

# Monitor
sudo radeontop
```

### Intel
```bash
# Install intel-gpu-tools
sudo apt install intel-gpu-tools

# Monitor
sudo intel_gpu_top
```

## Docker Support

```dockerfile
FROM ubuntu:22.04

# Install Vulkan
RUN apt update && apt install -y \
    vulkan-tools \
    mesa-vulkan-drivers \
    wget unzip

# Copy binaries
COPY qvac-linux-vulkan-x64-v1.0 /opt/qvac/

WORKDIR /opt/qvac
CMD ["/opt/qvac/bin/llama-cli", "--help"]
```

## Benchmark Your Hardware

```bash
# Inference benchmark
./bin/llama-cli \
  -m models/qwen3-1.7b-q8_0.gguf \
  -ngl 999 \
  -c 2048 \
  -s 42 \
  --temp 0 \
  -p "Tell me a joke" \
  --bench 5

# Training benchmark
time ./bin/llama-finetune-lora \
  -m models/qwen3-1.7b-q8_0.gguf \
  -f small_dataset.jsonl \
  -c 128 -b 128 -ub 128 -ngl 999 \
  --num-epochs 1
```

## Support

- 📚 [Main Documentation](../../README.md)
- 📊 [Benchmarks](../../docs/BENCHMARKS.md)
- 🐛 [Report Issues](https://github.com/akshaypn/qvac-finetune/issues)
- 💬 [Discussions](https://github.com/akshaypn/qvac-finetune/discussions)

## License

MIT License - See [LICENSE](../../LICENSE) for details.

