# Chapter 10 — Appendix

> *Quick reference material for when you know what you are looking for and just need the command, the fix, or the link.*

---

## A. Command Cheat Sheet

### GPU Status and Monitoring

```bash
# Snapshot of all GPU state
nvidia-smi

# Live view, refreshing every 2 seconds
watch -n 2 nvidia-smi

# Compact live dashboard with memory and utilization
nvidia-smi dmon -s pucvmet

# CSV output — useful for scripting and logging
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free,temperature.gpu \
    --format=csv,noheader

# Interactive GPU process manager (better than nvidia-smi for daily use)
pip install nvitop && nvitop

# Compact colorized GPU status
pip install gpustat && gpustat -i 1

# Kill a process holding VRAM (find PID in nvidia-smi first)
kill -9 <PID>
```

### CUDA and PyTorch Diagnostics

```python
import torch

torch.cuda.is_available()                              # True if GPU is ready
torch.cuda.device_count()                             # number of available GPUs
torch.cuda.get_device_name(0)                         # GPU model name
torch.version.cuda                                    # CUDA version PyTorch was built with
torch.cuda.memory_allocated() / 1e9                   # GB currently in use by tensors
torch.cuda.memory_reserved() / 1e9                    # GB held by PyTorch allocator
torch.cuda.get_device_properties(0).total_memory / 1e9  # total VRAM in GB
torch.cuda.empty_cache()                              # release cached memory
torch.cuda.synchronize()                              # wait for all GPU ops to finish
```

### Environment Setup

```bash
# Check installed CUDA toolkit version
nvcc --version

# Check maximum CUDA version supported by current driver
nvidia-smi | grep "CUDA Version"

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch with CUDA 12.1 via conda
conda install pytorch torchvision torchaudio \
    pytorch-cuda=12.1 -c pytorch -c nvidia

# Install CUDA toolkit on Ubuntu
sudo apt-get install cuda-toolkit-12-1
```

### Training Launch Commands

```bash
# DDP: single machine, 2 GPUs
torchrun --nproc_per_node=2 train.py

# DDP: single machine, 4 GPUs
torchrun --nproc_per_node=4 train.py

# DDP: 2 machines, 4 GPUs each (run on node 0)
torchrun --nproc_per_node=4 --nnodes=2 \
    --node_rank=0 --master_addr=192.168.1.10 --master_port=12355 train.py

# DDP: same command on node 1 (change --node_rank to 1)
torchrun --nproc_per_node=4 --nnodes=2 \
    --node_rank=1 --master_addr=192.168.1.10 --master_port=12355 train.py

# Enable synchronous CUDA execution for debugging
CUDA_LAUNCH_BLOCKING=1 python train.py
```

### Hugging Face and PEFT

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Load in 4-bit
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto")

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Show trainable parameters (after adding LoRA)
model.print_trainable_parameters()

# Save LoRA adapters
model.save_pretrained("./adapters")

# Load adapters back
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./adapters")

# Merge adapters into base model
merged = model.merge_and_unload()
```

### Remote Workflow (SSH / tmux / rsync)

```bash
# Connect to remote machine
ssh user@server-ip

# Start a named tmux session
tmux new-session -s training

# Detach from tmux session (process keeps running)
# Press: Ctrl+B, then D

# Reattach to existing session
tmux attach-session -t training

# List all tmux sessions
tmux ls

# Sync local code to remote (fast incremental)
rsync -avz --exclude='.git' --exclude='__pycache__' \
    ./project/ user@server:/home/user/project/

# Pull output files back
rsync -avz user@server:/home/user/project/outputs/ ./outputs/

# Forward Jupyter port over SSH
ssh -N -L 8888:localhost:8888 user@server-ip
# Then open: http://localhost:8888 in local browser
```

---

## B. Common Errors and Fixes

| Error | Meaning | Fix |
|---|---|---|
| `CUDA out of memory` | Batch too large; activations too large; optimizer states too large | Reduce batch size; add gradient checkpointing; use quantization; call `torch.cuda.empty_cache()` |
| `device-side assert triggered` | Invalid value sent to GPU kernel (label out of range, NaN input) | Run with `CUDA_LAUNCH_BLOCKING=1` to get full traceback |
| `Expected all tensors on same device` | A CPU tensor mixed into a GPU operation | Check `.device` on all tensors; add `.to(device)` to the offending tensor |
| `RuntimeError: CUDA error: no kernel image` | PyTorch not compiled for your GPU's compute capability | Reinstall PyTorch with the correct CUDA version for your hardware |
| `illegal memory access` | Operation on freed, invalid, or out-of-bounds memory | Print tensor shapes before the failing operation; check for use-after-free |
| `loss = nan` | Learning rate too high (exploding gradients) or FP16 overflow | Lower LR; add gradient clipping; switch FP16→BF16; add GradScaler |
| `NCCL error: unhandled system error` | Multi-GPU communication failure | Check shared memory (`/dev/shm`), firewall rules, and `ulimit -n` |
| `libcuda.so not found` | CUDA libraries not on the library path | `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` |
| `nvidia-smi: command not found` | Driver not installed or not on PATH | Install NVIDIA driver; check that `/usr/bin/nvidia-smi` exists |
| `torch.compile() errors` | Triton not installed or wrong PyTorch version | Requires PyTorch 2.0+; run `pip install triton` |
| TF: `No GPU found` | Using `tensorflow` instead of `tensorflow[and-cuda]` | `pip install tensorflow[and-cuda]` |
| `WinError 126` on Windows | Missing CUDA DLL in PATH | Add CUDA bin directory to Windows PATH |

---

## C. Curated Resources

### Official Documentation

The official docs should be your first stop for any specific API question. They are more reliable than Stack Overflow for current syntax and behaviour.

- **PyTorch Docs:** [pytorch.org/docs](https://pytorch.org/docs) — tutorials, API reference, and migration guides
- **Hugging Face Docs:** [huggingface.co/docs](https://huggingface.co/docs) — Transformers, Datasets, PEFT, TRL, Accelerate
- **CUDA Toolkit Documentation:** [docs.nvidia.com/cuda](https://docs.nvidia.com/cuda)
- **CUDA Compatibility Guide:** [docs.nvidia.com/deploy/cuda-compatibility](https://docs.nvidia.com/deploy/cuda-compatibility)
- **PyTorch GPU Performance Guide:** [pytorch.org/tutorials/recipes/recipes/tuning_guide.html](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

### Learning Resources

- **Fast.ai Practical Deep Learning for Coders:** [course.fast.ai](https://course.fast.ai) — arguably the best practical deep learning course; GPU-centric from day one
- **Andrej Karpathy — Neural Networks: Zero to Hero:** YouTube series building a language model from scratch; exceptional for building real understanding of transformers
- **Tim Dettmers' Blog:** [timdettmers.com](https://timdettmers.com) — the best independent source on GPU hardware recommendations for ML and on quantization techniques (bitsandbytes is his work)
- **Lilian Weng's Blog:** [lilianweng.github.io](https://lilianweng.github.io) — deep, well-sourced articles on LLMs, attention, and training techniques
- **Sebastian Raschka's Newsletter:** Practical ML Engineering — consistently useful for implementation-level understanding

### Tools and Libraries

- **nvitop:** [github.com/XuehaiPan/nvitop](https://github.com/XuehaiPan/nvitop) — interactive GPU monitor
- **Weights & Biases:** [wandb.ai](https://wandb.ai) — experiment tracking
- **vLLM:** [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) — production LLM serving
- **PEFT:** [github.com/huggingface/peft](https://github.com/huggingface/peft) — LoRA and other fine-tuning methods
- **bitsandbytes:** [github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — quantization
- **Flash Attention:** [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — memory-efficient attention kernel
- **Accelerate:** [github.com/huggingface/accelerate](https://github.com/huggingface/accelerate) — multi-GPU and mixed precision with minimal code changes
- **llama.cpp:** [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) — quantized inference on CPU and consumer GPUs

### Version Compatibility Reference

Before installing PyTorch for a new project or environment, check the compatibility matrix at:
[pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)

This page lists every PyTorch release with its corresponding supported CUDA versions and the exact pip/conda install command. It prevents the most common class of setup failures.

---

## D. Quick Reference: Memory Reduction Techniques

When you hit an OOM error, work through this list in order. Each technique reduces VRAM usage without changing the model architecture:

```
1. Reduce batch size (halving batch size ~halves activation memory)
2. Enable torch.no_grad() during evaluation (eliminates gradient storage)
3. Enable gradient checkpointing (60-70% less activation memory; +30% compute)
4. Switch to mixed precision BF16 (halves activation and gradient memory)
5. Use quantization: INT8 (~50% weight reduction) or INT4 (~75%)
6. Add gradient accumulation (same effective batch, less instantaneous VRAM)
7. Reduce sequence length (attention is O(n²) in memory)
8. Install Flash Attention (rewrites attention to be O(n) in memory)
9. Offload optimizer states to CPU (ZeRO stage 1/2 via DeepSpeed or Accelerate)
10. Use model parallelism (split model across multiple GPUs)
```

---

---

## Navigation

**[← Previous: Chapter 9 — Zero to Hero Roadmap](chapter-09-zero-to-hero-roadmap.md)**

**[← Back to Chapter 1](chapter-01-foundations.md)** — Start from the beginning.

This is the end of the guide. The best next step is to open a terminal, pick a project from Chapter 9, and start building.
