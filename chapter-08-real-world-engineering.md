# Chapter 8 — Real-World Engineering Tips

> *The difference between a working prototype and reliable production code is a collection of hard-won lessons. This chapter covers the engineering decisions and debugging skills that experienced ML engineers apply daily.*

---

## 8.1 Choosing the Right GPU

The best GPU for your work is determined almost entirely by one number: how much VRAM the model and training configuration require. Compute performance matters, but VRAM is the hard constraint that determines whether a job is even possible.

Start with the VRAM you need, then look at what fits your budget:

| Use Case | Minimum VRAM | Recommended GPU |
|---|---|---|
| Learning, small experiments ($<$1B params) | 6–8 GB | RTX 3060 / 3070 |
| Research, fine-tuning small models (1B–7B) | 16 GB | RTX 3090 / 4090 (24 GB) |
| LLM fine-tuning (7B–13B with LoRA) | 24 GB | RTX 3090 / 4090 |
| LLM fine-tuning (13B–34B with LoRA) | 40–48 GB | 2× RTX 3090 or RTX A6000 |
| Full fine-tuning of 7B+ models | 80 GB | A100 80GB (cloud) |
| LLM training from scratch | 160+ GB | Multiple H100s (cloud) |
| Inference only (7B, quantized) | 6 GB | RTX 3060 or better |

After VRAM, the next most important spec for training workloads is **memory bandwidth** — how fast data moves between VRAM and the compute units. The H100's 3.35 TB/s bandwidth is a large part of why it is so much faster than older cards, not just the raw TFLOPS.

For consumer GPU purchases, the secondhand market is worth serious consideration. RTX 3090s (24 GB, excellent for ML) are widely available used for around \$700–900, compared to \$1,500+ new. These cards run the same ML workloads as a brand-new one and are reliable for training. The main risk is buying from a crypto mining context — cards that ran at 100% utilization for two years are more likely to have degraded thermal paste or fan bearings.

> 💡 **Pro Tip:** Two RTX 3090s in a desktop (48 GB combined VRAM, ~\$1,600 total secondhand) will handle most LLM fine-tuning workloads and outperform a single A6000 for tasks that spread across GPUs. But for single-process jobs that need 48 GB in one contiguous pool, the A6000 wins.

---

## 8.2 Cost vs Performance Trade-offs

People new to cloud GPU often overcalibrate toward the most powerful hardware available. The H100 is the fastest GPU for most workloads, but it costs around \$4–5/hr on RunPod. An RTX 4090 on the same platform costs under \$1/hr. Whether the H100's speed advantage justifies the 5× cost depends entirely on how much of the speedup you actually capture.

For most fine-tuning and small training jobs, the bottleneck is not compute — it is data loading, the optimizer step, or the time you spend iterating on the code itself. Spending \$50 on an H100 for a job that an RTX 4090 could finish in the same wall-clock time at \$9 is a waste.

A rough decision framework:

- **RTX 4090 or similar consumer GPU:** Training runs under 10 hours, model fits in 24 GB VRAM. The daily workhorse for most ML engineers.
- **A100 40GB:** Model requires more than 24 GB, or you need ECC memory for very long training runs where silent bit errors matter.
- **A100 80GB / H100:** Training large models (13B+) or running experiments that genuinely require 80 GB of VRAM. The H100 also wins for workloads heavy with transformer attention (Flash Attention 2 saturates its bandwidth advantage).
- **Multi-GPU:** When the model does not fit in a single GPU, or when you need to run many parallel experiments simultaneously.

For **owned hardware vs cloud**, the rough breakeven calculation is: a consumer GPU costs around \$1,500–2,000 and provides roughly \$0.40/hr of equivalent cloud value. At 8 hours of use per day, that is approximately 500 days to break even. If you use it daily for more than 18 months, owned hardware wins economically. If you need GPUs for occasional short projects, cloud is more cost-efficient.

---

## 8.3 CUDA Errors Decoded

CUDA errors fall into a few recurring categories. Most of them have a clear cause and a direct fix once you know what to look for.

### Out of Memory (OOM)

```
RuntimeError: CUDA out of memory. Tried to allocate X GB
(GPU 0; Y GB total capacity; Z GB already allocated)
```

This is the most common error, and it almost always comes from one of three sources: the batch size is too large, gradient tracking is enabled during inference (wasting activation memory), or the model itself exceeds available VRAM.

```python
# OOM debugging checklist — try in order:

# 1. Are you tracking gradients during inference/evaluation?
with torch.no_grad():   # add this if missing
    output = model(inputs)

# 2. Can you reduce batch size?
# Halving batch size roughly halves activation memory.

# 3. Enable gradient checkpointing?
model.gradient_checkpointing_enable()

# 4. Use mixed precision?
with autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(inputs)

# 5. Is another process holding VRAM?
# Check with: nvidia-smi
# Kill lingering Python processes if found.

# 6. Did a previous run leak VRAM?
torch.cuda.empty_cache()
import gc; gc.collect()
```

### Device-Side Assert (Confusing Errors)

```
RuntimeError: CUDA error: device-side assert triggered
```

This error is deliberately vague because GPU operations are asynchronous — by the time the error surfaces in Python, the execution point that caused it is gone. The most common causes are class indices outside the valid range (e.g., passing a label of 10 to a model with only 10 classes, indexed 0–9) or NaN values in the input.

The fix for making this debuggable is a single environment variable:

```bash
# Forces CUDA to execute synchronously, so errors point to the exact Python line
CUDA_LAUNCH_BLOCKING=1 python train.py
```

```python
# Or set it at the top of your script during debugging
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```

With `CUDA_LAUNCH_BLOCKING=1`, the error message transforms from a cryptic async report into a proper Python traceback pointing at the line that caused the problem.

### Illegal Memory Access

```
RuntimeError: CUDA error: an illegal memory access was encountered
```

This usually means a tensor operation tried to access memory outside valid bounds — often caused by shape mismatches between tensors. Print the shapes of everything involved before the operation that triggers the error:

```python
# Debug shape issues before they become CUDA errors
print(f"Input shape:  {x.shape}")
print(f"Weight shape: {layer.weight.shape}")
print(f"Expected:     {(batch_size, seq_len, model_dim)}")

# Then run the operation
output = layer(x)
```

### Expected All Tensors on Same Device

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

A CPU tensor crept into a GPU operation. Find it by checking devices systematically:

```python
def find_cpu_tensors(module, prefix=""):
    """Walk a model's parameters and report any on CPU."""
    for name, param in module.named_parameters():
        if param.device.type == "cpu":
            print(f"  CPU param found: {prefix}{name} — shape: {param.shape}")

find_cpu_tensors(model)

# Also check your batch
print(f"X device: {X.device}, y device: {y.device}")
```

### Loss Is NaN or Inf

```
loss = tensor(nan)
```

NaN loss means the gradients or activations contain a value that overflowed or divided by zero. The typical causes are a learning rate that is too large (exploding gradients at the start of training), FP16 overflow (use BF16 or add `GradScaler`), or a numerical issue in the loss function (log of zero, division by a near-zero value).

```python
# Gradient clipping prevents exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check for NaN before the optimizer step
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN gradient in {name}")
```

---

## 8.4 Beginner Mistakes That Eat Debugging Time

These errors come up repeatedly across learners at every level. None of them are obvious from documentation, but all of them are fixable once you recognise the pattern.

### Forgetting `optimizer.zero_grad()`

PyTorch accumulates gradients by default — calling `.backward()` adds new gradients on top of whatever gradients are already stored in `param.grad`. If you forget to zero gradients at the start of each iteration, you are training on the sum of all previous gradients. The loss either diverges or behaves erratically in ways that are hard to trace back to this cause.

```python
# Correct loop structure — zero_grad() always comes first
optimizer.zero_grad()      # ← do not forget this
output = model(X)
loss   = criterion(output, y)
loss.backward()
optimizer.step()
```

### Not Switching Between `model.train()` and `model.eval()`

`Dropout` and `BatchNorm` behave differently depending on the mode the model is in. In training mode, dropout randomly zeros activations; in eval mode, it passes everything through. BatchNorm uses running statistics in eval mode and computes fresh statistics from the batch in training mode. Forgetting to switch produces inconsistent results that look like model bugs:

```python
# Training loop
model.train()
for X, y in train_loader:
    # ...

# Validation loop — always switch to eval
model.eval()
with torch.no_grad():
    for X_val, y_val in val_loader:
        # ...

# After validation — switch back for next training epoch
model.train()
```

### Calling `.item()` Inside a Tight Loop

`.item()` forces a CPU-GPU synchronization to pull a scalar value out of a GPU tensor. Calling it every iteration in a large training loop adds up to a significant overhead. The correct pattern is to accumulate the raw tensor and only call `.item()` when you actually need the value (e.g., for printing):

```python
# Slow: forces a GPU sync every iteration
for X, y in loader:
    loss = criterion(model(X), y)
    total_loss += loss.item()   # ← sync on every batch

# Better: accumulate tensor, sync once at the end of the epoch
running_loss = 0.0
for X, y in loader:
    loss = criterion(model(X), y)
    running_loss += loss          # stays on GPU; no sync

avg_loss = running_loss.item() / len(loader)   # single sync
```

### Ignoring `non_blocking=True`

When you have `pin_memory=True` in your DataLoader (which you should), `non_blocking=True` in the `.to(device)` call allows the CPU-to-GPU data transfer to happen asynchronously — the CPU can queue the next operation while the GPU is still receiving the current batch. Without it, the transfer is synchronous and the CPU waits:

```python
# Without non_blocking: CPU waits for transfer to complete
X = X.to(device)
y = y.to(device)

# With non_blocking: transfers queue asynchronously (requires pin_memory=True)
X = X.to(device, non_blocking=True)
y = y.to(device, non_blocking=True)
```

### Using DataParallel for Serious Training

As discussed in Chapter 5, `nn.DataParallel` is easy to add but creates an uneven memory load (GPU 0 gets extra work gathering gradients) and does not scale to multiple machines. It is fine for a quick experiment but should not be used for anything you care about. Set up DDP properly from the start.

> 🚀 **Best Practice:** Keep a personal checklist for new training jobs. Before starting, verify: data is on GPU, model is on GPU, `model.train()` is set, `zero_grad()` is in the loop, mixed precision is enabled, checkpoints are saving, and `nvidia-smi` shows the GPU is actually being used. Five minutes of verification saves hours of confusion.

---

## Chapter Summary

GPU selection comes down to VRAM first, then bandwidth, then compute. For most ML work on a budget, secondhand RTX 3090s offer exceptional value. Cloud GPUs are for workloads that genuinely exceed what local hardware can do — not just for the sake of having access to a fancier card.

The CUDA error messages that seem most confusing (`device-side assert triggered`, `illegal memory access`) are almost always manageable once you know to run with `CUDA_LAUNCH_BLOCKING=1`. That single environment variable converts opaque async errors into debuggable Python tracebacks.

The beginner mistakes are consistent across learners: forgotten `zero_grad()`, unchecked model/eval mode, unnecessary `.item()` calls in tight loops, and silent CPU fallback. A short checklist before starting any training run prevents most of them.

---

*Next: Chapter 9 — Zero to Hero Roadmap. A concrete week-by-week learning path, the recommended tool stack, and mini-projects that build real GPU programming intuition.*
