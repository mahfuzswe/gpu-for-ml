# Chapter 4 — Performance Optimization

> *Getting code to run on GPU is the first step. Getting it to run fast is where most of the real engineering lives. This chapter covers the techniques that deliver the biggest gains — in order of impact.*

---

## Why Optimization Matters More Than Hardware

Before diving into techniques, it is worth understanding what "optimization" actually means in the GPU context. A poorly optimized training loop on an RTX 4090 can easily run slower than a well-optimized loop on an RTX 3070. The hardware matters, but how you use it matters more.

Most inefficiency comes from one of three sources: the GPU is sitting idle waiting for data, the GPU is doing unnecessary work in a high-precision format when lower precision is sufficient, or the training configuration is wasting memory on activations that do not need to be stored. Each technique in this chapter targets one of these.

The order here is roughly by impact-per-effort. Start with batch size, then add mixed precision, then optimize your DataLoader. Each step builds on the previous.

---

## 4.1 Batch Size Tuning

Batch size is the most direct knob for controlling GPU utilization. A batch size of 4 leaves most of the GPU's parallel execution units idle; a batch size of 512 keeps them busy. But there are tradeoffs in both directions.

**Larger batches:** better GPU utilization, smoother gradient estimates, fewer optimizer steps per epoch. However, very large batches can hurt generalization — the gradients become too averaged and the optimizer tends to find sharper, less robust minima.

**Smaller batches:** more frequent updates, noisier gradients that can actually help generalization, but poor GPU utilization and slower wall-clock training.

A practical approach is to find the largest batch size your VRAM allows, then tune the learning rate to compensate (a common rule of thumb: scale learning rate linearly with batch size).

```python
import torch
import torch.nn as nn

def find_max_batch_size(model, input_shape, device, start=32):
    """
    Doubles batch size until an OOM error occurs,
    then returns the last safe size.
    """
    model = model.to(device)
    batch_size = start

    for attempt in range(12):
        try:
            # Create a dummy batch and run a forward pass
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            _ = model(dummy_input)
            torch.cuda.empty_cache()
            print(f"  batch_size={batch_size:>5} → OK")
            batch_size *= 2

        except torch.cuda.OutOfMemoryError:
            print(f"  batch_size={batch_size:>5} → OOM")
            torch.cuda.empty_cache()
            # Return one step back — leave ~20% VRAM headroom for activations
            return batch_size // 4

    return batch_size

model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))
safe_bs = find_max_batch_size(model, (512,), device=torch.device("cuda"))
print(f"\nRecommended batch size: {safe_bs}")
```

> 💡 **Pro Tip:** Always use batch sizes that are powers of 2 (32, 64, 128, 256, 512...). CUDA memory allocators and GEMM (matrix multiplication) kernels are tuned for these dimensions. A batch size of 100 is often measurably slower than 128.

---

## 4.2 Mixed Precision Training

This is consistently the highest-impact optimization for most workloads. Mixed precision uses 16-bit floats for the bulk of computation while keeping 32-bit for accumulations that need precision. The result is roughly 2× faster training on Tensor Core hardware and about 40–50% less VRAM for activations.

### FP16 vs BF16 — Which One to Use

Both are 16-bit formats, but they distribute bits differently:

```
FP32  [ 1 sign | 8 exponent | 23 mantissa ]   range: ~±3.4×10³⁸
FP16  [ 1 sign | 5 exponent | 10 mantissa ]   range: ~±65,504   ← narrow; overflows easily
BF16  [ 1 sign | 8 exponent |  7 mantissa ]   range: ~±3.4×10³⁸ ← same range as FP32
```

BF16 has the same exponent width as FP32, so it rarely overflows or produces NaN values. FP16 is more likely to overflow during training — especially with large gradients — which is why it requires a loss scaler.

**Use BF16** if your GPU supports it (A100, H100, RTX 3090+, RTX 40-series, Apple M-series). It is simpler and more stable. **Use FP16** if you are on older hardware (V100, T4, RTX 20-series) that does not support BF16.

### Enabling Mixed Precision with `torch.amp`

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

device    = torch.device("cuda")
model     = nn.Linear(1024, 512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# GradScaler: compensates for FP16's narrow range by scaling
# the loss upward before backward, then unscaling before the optimizer step.
# If you use BF16, you technically do not need it, but it is harmless to include.
scaler = GradScaler()

for X, y in dataloader:
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()

    # autocast: PyTorch automatically selects FP16 or FP32 for each operation.
    # Matrix multiplies → FP16 (Tensor Core).
    # Layer norm, softmax, loss → FP32 (needs precision).
    with autocast(device_type="cuda", dtype=torch.float16):
        output = model(X)
        loss   = criterion(output, y)

    # Scale the loss before backward to prevent gradient underflow in FP16
    scaler.scale(loss).backward()

    # Unscale gradients, then check for inf/nan before stepping
    scaler.step(optimizer)
    scaler.update()   # adjusts scale factor for next iteration
```

For BF16 on modern hardware, the code simplifies further — no scaler needed:

```python
# BF16: stable range means no loss scaling required
with autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(X)
    loss   = criterion(output, y)

loss.backward()
optimizer.step()
optimizer.zero_grad()
```

> 🚀 **Best Practice:** Add mixed precision first before any other optimization. It is low-risk, high-reward, and takes about five lines of code changes. On a GPU with Tensor Cores (any RTX card or data center GPU), the speedup is almost always between 1.5× and 3×.

---

## 4.3 Memory Optimization

Running out of VRAM is the most common hard stop in GPU training. These techniques let you train larger models or larger batches on the same hardware.

### Release Unused Tensors

PyTorch's memory allocator holds on to freed memory in a cache to avoid the overhead of allocating fresh GPU memory on every operation. This is efficient, but it means memory usage can look higher than the actual live tensors warrant. To release that cache back to the OS:

```python
import torch, gc

def free_gpu_memory():
    """Release PyTorch's memory cache and run garbage collection."""
    gc.collect()                  # Python GC first
    torch.cuda.empty_cache()      # return cached-but-free VRAM to OS

# After finishing inference or evaluation:
with torch.no_grad():
    predictions = model(val_inputs)

del val_inputs, predictions   # drop references
free_gpu_memory()
```

### Gradient Checkpointing

This is a memory-compute tradeoff: instead of storing all intermediate activations during the forward pass (needed for backprop), only selected checkpoints are stored. The discarded activations are recomputed on-the-fly during the backward pass.

The result is typically a 60–70% reduction in activation memory, at the cost of about 30% extra compute time. For large models where VRAM is the bottleneck, this trade is almost always worth it.

```python
from torch.utils.checkpoint import checkpoint
import torch.nn as nn

class CheckpointedTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def _attn_block(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm1(x + attn_out)

    def forward(self, x):
        # checkpoint tells PyTorch: do not store activations from _attn_block.
        # Recompute them during backward instead.
        x = checkpoint(self._attn_block, x, use_reentrant=False)
        x = self.norm2(x + self.ff(x))
        return x
```

For Hugging Face Transformers, enabling gradient checkpointing is a single method call:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
model.gradient_checkpointing_enable()   # done — all transformer blocks use checkpointing
model = model.cuda()
```

### Always Disable Gradients During Inference

Gradient tracking consumes roughly twice as much memory during the forward pass because PyTorch stores activations for later backpropagation. During evaluation or inference, you never run `backward()`, so those activations are wasted. Turn tracking off:

```python
# torch.no_grad(): disables gradient tracking
# Use for validation loops and evaluation
model.eval()
with torch.no_grad():
    val_loss = 0.0
    for X_val, y_val in val_loader:
        X_val, y_val = X_val.to(device), y_val.to(device)
        output   = model(X_val)
        val_loss += criterion(output, y_val).item()

# torch.inference_mode(): even more aggressive — disables view tracking too
# Use for pure inference (no need to call .backward() after)
with torch.inference_mode():
    predictions = model(test_inputs)
```

---

## 4.4 DataLoader Optimization

A GPU running at 10% utilization while waiting for data is a GPU you are paying for but not using. DataLoader configuration is often the cheapest performance win available — and it is almost always worth checking before touching any other part of the training loop.

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=256,

    # num_workers: how many CPU processes load data in parallel.
    # With num_workers=0 (the default), data loading happens on the main thread
    # and the GPU starves while it waits. A good starting point is 4.
    # More is not always better — tune empirically for your dataset.
    num_workers=4,

    # pin_memory=True: copies data into pinned (page-locked) CPU memory,
    # which allows DMA (direct memory access) transfers to GPU — faster than
    # copying from pageable memory.
    pin_memory=True,

    # persistent_workers=True: keeps worker processes alive between epochs.
    # Without this, PyTorch tears down and rebuilds workers every epoch,
    # which wastes 5–10 seconds per epoch on datasets with complex augmentation.
    persistent_workers=True,

    # prefetch_factor: each worker pre-loads this many batches ahead of the
    # GPU's current position. 2 is a safe default.
    prefetch_factor=2,

    shuffle=True,
    drop_last=True,   # avoids a final small batch that could cause shape issues
)
```

| Setting | Default | Recommended | Why |
|---|---|---|---|
| `num_workers` | 0 | 4–8 | Parallel loading prevents GPU starvation |
| `pin_memory` | False | True | Faster CPU→GPU memory transfers |
| `persistent_workers` | False | True | No worker restart cost between epochs |
| `prefetch_factor` | 2 | 2–4 | Pre-loads batches while GPU runs |

> 💡 **Pro Tip:** If you are not sure whether your bottleneck is DataLoader or compute, watch `GPU-Util` in `nvidia-smi` during training. If it fluctuates between ~0% and ~90%, your DataLoader is too slow to keep the GPU consistently fed. If it stays above 80%, your DataLoader is fine.

---

## 4.5 Gradient Accumulation

Gradient accumulation lets you simulate a larger effective batch size without allocating the full batch in VRAM simultaneously. Instead of doing one forward-backward pass on 256 samples, you do 8 passes of 32 samples each and accumulate gradients before calling `optimizer.step()`.

This is especially useful when fine-tuning large models where even a batch size of 32 may push VRAM limits.

```python
# Target: effective batch size of 256
# Actual VRAM batch size: 32
# Accumulation steps: 8 (8 × 32 = 256)

accumulation_steps = 8
optimizer.zero_grad()

for i, (X, y) in enumerate(loader):
    X, y = X.to(device), y.to(device)

    with autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(X)
        # Divide loss by accumulation_steps so that the accumulated gradient
        # magnitude is the same as it would be for a true batch of 256.
        loss = criterion(output, y) / accumulation_steps

    scaler.scale(loss).backward()

    # Only step and zero gradients every N batches
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # Handle the final incomplete accumulation at end of epoch
    if i == len(loader) - 1 and (i + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

> ⚠️ **Common Mistake:** Forgetting to divide the loss by `accumulation_steps`. Without this, the accumulated gradient is `N` times larger than it should be, effectively multiplying your learning rate by `N`. Training will diverge.

---

## 4.6 Profiling Your Training Loop

Before spending time optimizing, profile. The assumption about where time is spent is wrong more often than not — profiling replaces guessing with measurements.

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    # wait=1: skip 1 step (warm-up)
    # warmup=1: record but discard 1 step (JIT compilation happens here)
    # active=5: record 5 steps of real data
    schedule=profiler.schedule(wait=1, warmup=1, active=5),
    on_trace_ready=profiler.tensorboard_trace_handler("./log/profiler"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        with autocast(device_type="cuda"):
            output = model(X)
            loss   = criterion(output, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        prof.step()   # tell the profiler this step is done

        if step >= 8:
            break

# Print a summary sorted by total CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

```bash
# Launch TensorBoard to view the full trace interactively
pip install tensorboard
tensorboard --logdir=./log/profiler
# Open http://localhost:6006 → click "PyTorch Profiler" tab
```

The profiler will show you exactly which operations take the most CUDA time, what the CPU-GPU overlap looks like, and where memory allocations occur. Common findings: the data loading pipeline accounts for 30–50% of wall time on unoptimized setups; the first batch is 5–10× slower than subsequent batches due to kernel compilation.

---

## Putting It All Together

Here is a training loop that applies every technique from this chapter:

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

device = torch.device("cuda")

# DataLoader with all optimizations
loader = DataLoader(
    dataset, batch_size=128, shuffle=True,
    num_workers=4, pin_memory=True,
    persistent_workers=True, drop_last=True,
)

model     = YourModel().to(device)
model     = torch.compile(model)      # PyTorch 2.0+: free speedup from kernel fusion
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler    = GradScaler()
criterion = nn.CrossEntropyLoss()

ACCUM_STEPS = 4   # effective batch = 128 × 4 = 512

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    for i, (X, y) in enumerate(loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            out  = model(X)
            loss = criterion(out, y) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

> 🚀 **Best Practice:** Notice `non_blocking=True` in `.to(device)` calls. When `pin_memory=True` in the DataLoader, `non_blocking=True` lets the data transfer happen asynchronously — the CPU moves on to the next step while the GPU transfer runs in the background, reducing idle time between batches.

---

## Chapter Summary

Performance optimization follows a natural priority order. Start with batch size (fill the GPU), then add mixed precision (activate Tensor Cores), then fix your DataLoader (stop the GPU from idling), then add gradient accumulation (simulate larger batches without extra VRAM). Profile before spending time on anything deeper than these four.

A training loop with all four techniques typically runs 3–8× faster than a naive baseline on the same hardware. The biggest single improvement is almost always the DataLoader — it is easy to configure and is underestimated by most beginners.

---

## Navigation

**[← Previous: Chapter 3 — First Practical GPU Usage](chapter-03-first-gpu-usage.md)**

**[➜ Next: Chapter 5 — Deep Learning Workloads](chapter-05-deep-learning-workloads.md)** — Putting these techniques into full CNN and Transformer training pipelines, handling large datasets, and scaling to multiple GPUs.
