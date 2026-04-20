# Chapter 3 — First Practical GPU Usage

> *Setup is done. Now you actually use the thing. This chapter covers the fundamental patterns that appear in almost every GPU training script you will ever write.*

---

## 3.1 Device Placement — The Core Concept

PyTorch follows one simple rule: **operations happen where the data lives**. A tensor on the CPU runs on the CPU. A tensor on the GPU runs on the GPU. If you try to mix them in a single operation, PyTorch raises an error immediately.

This means the first thing every GPU training script does is establish a *device* variable and make sure all data moves to it:

```python
import torch

# Check what is available and set the target device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")   # cuda on a GPU machine, cpu as fallback

# Create a tensor directly on the target device
x = torch.randn(1000, 1000, device=device)

# Or move an existing tensor from CPU to GPU
cpu_tensor = torch.randn(500, 500)          # lives on CPU
gpu_tensor = cpu_tensor.to(device)          # copy to GPU
# cpu_tensor still exists on CPU; .to() creates a new tensor

# Check where a tensor lives
print(gpu_tensor.device)   # cuda:0
```

The pattern `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` is worth memorising. It makes your code portable: it runs on GPU when one is available and falls back to CPU otherwise.

### Moving Models to GPU

A PyTorch model is a collection of parameter tensors. Calling `.to(device)` moves all of them — weights, biases, buffers — to the target device in one go.

```python
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A simple two-layer network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleNet().to(device)   # all parameters now live on GPU

# Verify
print(next(model.parameters()).device)   # cuda:0
```

> ⚠️ **Common Mistake:** Moving the model to GPU but forgetting to move the input data. This causes `RuntimeError: Expected all tensors to be on the same device`. Every tensor involved in a forward pass must be on the same device as the model.

---

## 3.2 A Complete Training Loop on GPU

Here is a minimal but complete training loop. Every real training script you write will follow this skeleton — the specifics change, but the structure does not.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Synthetic dataset for demonstration ───────────────────────────────
X = torch.randn(2000, 20)         # 2000 samples, 20 features
y = torch.randint(0, 3, (2000,))  # 3-class labels
dataset = TensorDataset(X, y)
loader  = DataLoader(dataset, batch_size=64, shuffle=True)

# ── Model ─────────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(20, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
).to(device)                      # model lives on GPU

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── Training loop ─────────────────────────────────────────────────────
for epoch in range(20):
    model.train()                  # activates dropout, batchnorm training behavior
    running_loss = 0.0

    for X_batch, y_batch in loader:
        # Move each batch to the same device as the model — critical step
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()      # clear gradients from previous step
        output = model(X_batch)    # forward pass — runs entirely on GPU
        loss   = criterion(output, y_batch)
        loss.backward()            # compute gradients via backprop
        optimizer.step()           # update weights

        running_loss += loss.item()   # .item() pulls the scalar to CPU

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch + 1:>3} | loss: {avg_loss:.4f}")
```

Four steps repeat inside every training loop, and they never change: `zero_grad()`, `forward pass`, `backward()`, `step()`. Understand what each one does and you understand training.

> ⚠️ **Common Mistake:** Forgetting `optimizer.zero_grad()`. PyTorch accumulates gradients by default — it adds new gradients on top of old ones every time `backward()` is called. If you forget to zero them at the start of each iteration, gradients from previous batches corrupt the current update and training diverges in unpredictable ways.

---

## 3.3 Monitoring GPU Usage with nvidia-smi

`nvidia-smi` is the first tool you should open whenever you start a training run. It shows you what the GPU is actually doing — not what you think it is doing.

```bash
# One-time snapshot of current GPU state
nvidia-smi

# Live view, refreshing every second
watch -n 1 nvidia-smi

# Compact live dashboard — useful during training
nvidia-smi dmon -s pucvmet

# Query specific fields for scripting
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free,temperature.gpu \
    --format=csv,noheader
```

The output of a basic `nvidia-smi` call looks like this:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 545.23     Driver Version: 545.23     CUDA Version: 12.3        |
|-------------------------------+----------------------+----------------------+
| GPU  Name         Persistence | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf   Pwr:Usage   |    Memory-Usage      | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 4090     Off  | 00000000:01:00.0 Off |                  N/A |
| 23%   42C    P2   180W / 450W |  12543MiB / 24564MiB |     87%    Default   |
+-----------------------------------------------------------------------------+
```

The four numbers to watch during training are: **GPU-Util** (how hard the GPU is working), **Memory-Usage** (VRAM consumed), **Pwr:Usage** (watts drawn — near TDP means full load), and **Temp** (above 85°C means the card may throttle itself).

| Field | What to Watch For |
|---|---|
| GPU-Util % | Near 0% during training → your data pipeline is the bottleneck, not compute |
| Memory-Usage | Near 100% → an OOM error is coming; reduce batch size or enable gradient checkpointing |
| Temp (°C) | Above 85°C → GPU throttles clock speed to protect hardware |
| Power (W) | Near TDP → GPU is at full load; well below TDP → under-utilised |

> 💡 **Pro Tip:** If `GPU-Util` stays near 0% while your training loop is running, the GPU is idling while it waits for the CPU to prepare the next batch. The fix is almost always increasing `num_workers` in your DataLoader (covered in Chapter 4).

### Better Tools: nvitop and gpustat

`nvidia-smi` works, but there are better alternatives for daily use:

```bash
# gpustat: compact, colorised, shows per-process VRAM usage
pip install gpustat
gpustat -i 1        # refresh every second

# nvitop: interactive process manager, like htop but for GPUs
pip install nvitop
nvitop
```

`nvitop` in particular is worth installing permanently. It shows per-process GPU and VRAM usage, live graphs of utilization, and lets you kill processes directly from the interface.

### Monitoring Memory from Inside Python

Sometimes you want to query GPU memory programmatically — for example, to log VRAM usage at checkpoints or print it after each epoch:

```python
import torch

def gpu_memory_report():
    """Print current GPU memory usage in a readable format."""
    if not torch.cuda.is_available():
        print("No CUDA device.")
        return

    props     = torch.cuda.get_device_properties(0)
    total_mb  = props.total_memory   / 1024**2
    alloc_mb  = torch.cuda.memory_allocated()  / 1024**2
    reserv_mb = torch.cuda.memory_reserved()   / 1024**2

    print(f"  GPU:       {props.name}")
    print(f"  Total:     {total_mb:,.0f} MB")
    print(f"  Allocated: {alloc_mb:,.1f} MB  (tensors actually in use)")
    print(f"  Reserved:  {reserv_mb:,.1f} MB  (held by PyTorch allocator)")
    print(f"  Free:      {total_mb - reserv_mb:,.1f} MB")

gpu_memory_report()
```

---

## 3.4 Debugging: When Code Silently Runs on CPU

This is one of the most common beginner frustrations: the code runs, produces output, but is 50–200× slower than expected because everything is running on CPU. There is no error — it just never moved to the GPU.

The most reliable way to catch this is to explicitly check where your data and model parameters live:

```python
def verify_on_gpu(model, sample_input, device):
    """
    Confirm that both model parameters and input data
    are on the expected device. Print a warning for any CPU tensors found.
    """
    all_good = True

    # Check every parameter in the model
    for name, param in model.named_parameters():
        if param.device.type != device.type:
            print(f"  ⚠ Parameter on wrong device: {name} → {param.device}")
            all_good = False

    # Check input tensor
    if sample_input.device.type != device.type:
        print(f"  ⚠ Input tensor on wrong device: {sample_input.device}")
        all_good = False

    if all_good:
        print(f"  ✓ Model and inputs confirmed on {device}")

device = torch.device("cuda")
verify_on_gpu(model, X_batch, device)
```

Another useful technique during development is adding a hard assertion at the start of your training loop. It makes mismatched devices fail fast with a clear message instead of silently running on CPU:

```python
for X_batch, y_batch in loader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    # Add during debugging; remove before long training runs
    assert X_batch.is_cuda, f"X_batch is on CPU! Check your .to(device) call."
    assert next(model.parameters()).is_cuda, "Model is on CPU!"

    # ... rest of loop
```

---

## 3.5 GPU Timing — Why You Must Synchronize

GPU operations are **asynchronous** by default. When you call `model(x)` or `loss.backward()`, Python submits those operations to a GPU work queue and immediately moves on to the next line — it does not wait for the GPU to finish. This is intentional (it avoids wasting CPU time waiting), but it makes timing tricky.

If you measure elapsed time without synchronizing, you are measuring how long it took to *submit* the work, not how long the work actually took:

```python
import torch
import time

device = torch.device("cuda")
A = torch.randn(8192, 8192, device=device)
B = torch.randn(8192, 8192, device=device)

# ── WRONG: measures queue submission time, not actual compute ─────────
start = time.time()
C = A @ B                         # submitted to GPU queue — returns immediately
wrong_time = time.time() - start  # almost always shows < 0.001s regardless of work
print(f"Wrong measurement: {wrong_time:.6f}s")

# ── CORRECT: synchronize before and after to measure actual compute ───
torch.cuda.synchronize()          # wait for all pending GPU ops
start = time.time()
C = A @ B
torch.cuda.synchronize()          # wait for this op to complete
correct_time = time.time() - start
print(f"Correct measurement: {correct_time:.4f}s")   # reflects real GPU time
```

For production code, you would not add `synchronize()` calls everywhere — they add overhead by breaking the pipeline between CPU and GPU. But during benchmarking, profiling, or whenever you want to understand how long something actually takes on the GPU, they are essential.

---

## Chapter Summary

The foundation of every GPU training script is device management: detect the device once, move the model once, move each batch inside the loop. Three things must always be on the same device for an operation to work — model parameters, input data, and labels.

Monitor your training with `nvidia-smi` or `nvitop`. GPU-Util near 0% means the GPU is waiting for data, not computing — that is a DataLoader problem, not a model problem. Memory near full means an OOM error is approaching.

When debugging silent CPU fallback, check `tensor.device` and `next(model.parameters()).device` explicitly. When timing GPU operations, synchronize before and after measurement or the numbers will be meaningless.

---

*Next: Chapter 4 — Performance Optimization. Making your training loops faster: mixed precision, batch tuning, DataLoader configuration, gradient accumulation, and profiling.*
