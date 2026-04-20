# Chapter 2 — Environment Setup

> *Getting your GPU environment working correctly is half the battle. Most beginners spend more time here than anywhere else. This chapter cuts through the confusion with a direct, step-by-step process.*

---

## 2.1 Understand the Version Dependency Chain First

Before you install anything, understand what depends on what. Getting this wrong is the root cause of most setup failures — people install the newest PyTorch, then the newest CUDA, without checking whether those two versions are even compatible.

The stack looks like this:

```
   Your Code  (Python / PyTorch / TensorFlow)
          ↓
   Deep Learning Framework  (PyTorch or TF)
          ↓
   CUDA Toolkit  (cuBLAS, cuDNN, NCCL, etc.)
          ↓
   NVIDIA GPU Driver
          ↓
   Physical GPU Hardware
```

Each layer depends on the one below it. The **GPU driver** sets the ceiling — it determines the maximum CUDA version your system can support. The **CUDA toolkit** and **cuDNN** must be mutually compatible. And your **PyTorch version** must be compiled against the CUDA version you have installed.

The practical rule: always start from the bottom. Check your driver first. Then choose a compatible CUDA version. Then install the matching PyTorch.

> ⚠️ **Common Mistake:** Installing the newest PyTorch, then trying to match CUDA to it, without first checking whether your current driver supports that CUDA version. Always verify `nvidia-smi` before anything else.

---

## 2.2 Local Setup on Linux

Linux is the preferred platform for ML development. The tooling is more mature, most documentation assumes Linux, and GPU drivers tend to be more stable than their Windows counterparts.

### Step 1 — Check your GPU and current driver

```bash
# List NVIDIA GPU hardware detected by the system
lspci | grep -i nvidia

# Check if a driver is already installed
nvidia-smi
```

If `nvidia-smi` runs successfully, read two numbers from its output carefully: **Driver Version** (top-right area) and **CUDA Version** (also top-right). That CUDA version shown by `nvidia-smi` is the *maximum* CUDA your current driver supports — not the version actually installed.

### Step 2 — Install the NVIDIA Driver

```bash
# Ubuntu's automatic driver detection (easiest for beginners)
sudo ubuntu-drivers autoinstall
sudo reboot

# Or install a specific driver version if you need control
sudo apt install nvidia-driver-545
sudo reboot

# After rebooting, confirm it worked
nvidia-smi
```

### Step 3 — Install the CUDA Toolkit

Go to [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads), select your operating system and architecture, and the page will generate the exact install commands for you. Below is an example for CUDA 12.1 on Ubuntu 22.04:

```bash
# Download and register the CUDA package repository key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update package list and install
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1

# Add CUDA to your PATH — add these two lines to ~/.bashrc
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Reload shell config and verify
source ~/.bashrc
nvcc --version   # should report 12.1
```

### Step 4 — Install cuDNN

cuDNN is NVIDIA's library of GPU-accelerated primitives for deep learning — convolutions, activations, pooling, and so on. PyTorch bundles its own cuDNN, so if you only use PyTorch, you can skip this step. If you use TensorFlow with a manually installed CUDA, you need cuDNN.

```bash
sudo apt-get install libcudnn8 libcudnn8-dev
```

### Step 5 — Install PyTorch with GPU Support

Always use the official command generator at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). It prevents version mismatches. Here is what the output looks like for CUDA 12.1:

```bash
# With pip (recommended — use inside a virtual environment)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# With conda
conda install pytorch torchvision torchaudio \
    pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## 2.3 Local Setup on Windows

Windows works, but has meaningfully more friction. The recommended approach is **WSL2** (Windows Subsystem for Linux) running Ubuntu 22.04. WSL2 gives you a proper Linux environment while the NVIDIA GPU driver is shared from the Windows side — so you get the compatibility of Linux without having to dual-boot.

```powershell
# Run in PowerShell as Administrator
wsl --install
wsl --set-default-version 2
wsl --install -d Ubuntu-22.04
```

Once WSL2 is running, follow the Linux steps above exactly — except for one critical rule:

> ⚠️ **Common Mistake:** Installing a Linux NVIDIA driver *inside* WSL2. WSL2 uses the Windows NVIDIA driver to expose CUDA to the Linux environment. Installing a driver inside WSL2 will break GPU access. Inside WSL2, install only the CUDA toolkit — never the driver.

For native Windows without WSL2: download the CUDA toolkit `.exe` installer from NVIDIA's website and use the graphical wizard. Then install PyTorch as shown in Step 5 above.

---

## 2.4 Verifying Your GPU Setup

Run this script every time you configure a new environment. It checks each layer of the stack and tells you immediately if something is wrong.

```python
import torch

# ── Basic checks ──────────────────────────────────────────────────────
print("PyTorch version:        ", torch.__version__)
print("CUDA available:         ", torch.cuda.is_available())   # Must be True
print("CUDA version (PyTorch): ", torch.version.cuda)
print("GPU count:              ", torch.cuda.device_count())

# ── GPU details ───────────────────────────────────────────────────────
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU name:      {props.name}")
    print(f"VRAM total:    {props.total_memory / 1e9:.1f} GB")
    print(f"CUDA capability: {props.major}.{props.minor}")

    # ── Smoke test: allocate a tensor on GPU ──────────────────────────
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"\nTest tensor device: {x.device}")           # should say cuda:0
    print(f"Test tensor value:  {x.sum().item()}")       # should be 6.0

    # ── Memory info ───────────────────────────────────────────────────
    allocated = torch.cuda.memory_allocated() / 1e6
    total     = props.total_memory / 1e9
    print(f"\nVRAM in use:   {allocated:.1f} MB")
    print(f"VRAM total:    {total:.1f} GB")
else:
    print("\nCUDA not available — check driver and PyTorch install.")
```

If `torch.cuda.is_available()` returns `False`, the problem is almost always one of three things: the driver is not installed, the CUDA version does not match the PyTorch build, or the PyTorch install itself used the CPU-only wheel by mistake.

---

## 2.5 TensorFlow GPU Setup

TensorFlow handles GPU setup differently from PyTorch. Modern TensorFlow (2.x) bundles compatible CUDA and cuDNN libraries directly inside the pip package — so you do not need a separate CUDA toolkit installation if you use the bundled version.

```bash
# The [and-cuda] extra includes bundled CUDA/cuDNN
pip install tensorflow[and-cuda]

# Verify GPU is detected
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs found: {len(gpus)}')
for g in gpus:
    print(' ', g)
"
```

> 💡 **Pro Tip:** If you need both PyTorch and TensorFlow in the same environment, use separate conda environments. Their CUDA dependencies can conflict if installed together without careful version pinning.

---

## 2.6 Managing Environments

Different projects often need different PyTorch versions. Keeping them isolated prevents version conflicts that are genuinely painful to debug.

```bash
# Create a named conda environment for a project
conda create -n gpu-ml python=3.11
conda activate gpu-ml

# Install PyTorch inside the environment
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Or with venv + pip
python -m venv .venv
source .venv/bin/activate       # on Linux/macOS
# .venv\Scripts\activate        # on Windows
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> 🚀 **Best Practice:** Always work inside a virtual environment. Create one per project, name it descriptively, and commit a `requirements.txt` or `environment.yml` so anyone (including future you) can reproduce the setup.

---

## 2.7 Common Setup Failures and Fixes

| Error / Symptom | Most Likely Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` → False | Driver missing or PyTorch CPU-only build | Check `nvidia-smi`; reinstall PyTorch with correct CUDA flag |
| `CUDA version mismatch` | PyTorch CUDA ≠ installed CUDA | Run `nvcc --version` and `nvidia-smi`; match PyTorch wheel to CUDA |
| `libcuda.so not found` | CUDA libs not on LD_LIBRARY_PATH | Add CUDA lib64 to `LD_LIBRARY_PATH`; run `sudo ldconfig` |
| `nvidia-smi not found` | Driver not installed or PATH issue | Reinstall driver; check `/usr/bin/nvidia-smi` exists |
| OOM error immediately on startup | Another process is holding VRAM | Run `nvidia-smi`; kill the process using VRAM |
| WSL2 GPU not detected | Linux driver installed inside WSL2 | Remove Linux driver; only install CUDA toolkit, not driver |
| TF: `No GPU found` | Using tensorflow (not tensorflow[and-cuda]) | Reinstall with `pip install tensorflow[and-cuda]` |

---

## Chapter Summary

The setup process has a strict order that you cannot shortcut: check your driver first, then choose a compatible CUDA version, then install matching PyTorch. Every setup failure traces back to a break somewhere in this dependency chain.

On Linux, follow the five steps in sequence. On Windows, WSL2 with Ubuntu 22.04 is the path of least resistance — just remember never to install the GPU driver inside the WSL2 environment.

Once `torch.cuda.is_available()` returns `True`, your GPU name appears in the verification script, and your test tensor shows `device: cuda:0` — you are ready to write real code.

---

*Next: Chapter 3 — First Practical GPU Usage. Moving tensors to GPU, writing a complete training loop, and monitoring what the GPU is actually doing.*
