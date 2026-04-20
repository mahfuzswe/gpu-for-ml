# Chapter 5 — Deep Learning Workloads on GPU

> *Individual techniques are useful. Seeing them assembled into real workloads is where understanding becomes practical skill. This chapter walks through the two major deep learning architectures — CNNs and Transformers — as complete GPU workflows.*

---

## 5.1 CNN Training Workflow

Convolutional Neural Networks were the workload that originally demonstrated GPU acceleration for deep learning. The training workflow is mature and well-understood, making CNNs an excellent reference point before moving on to more complex architectures.

The example below trains a ResNet-18 on CIFAR-10, incorporates every optimization technique from Chapter 4, and demonstrates `torch.compile` — a PyTorch 2.0 feature that compiles your model into optimized low-level kernels for a free speedup.

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data pipeline ─────────────────────────────────────────────────────
# Random augmentation during training prevents overfitting.
# Normalization uses CIFAR-10's channel statistics.
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform)
val_set   = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=val_transform)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=256, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = torch.utils.data.DataLoader(
    val_set,   batch_size=512, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True)

# ── Model ─────────────────────────────────────────────────────────────
model = torchvision.models.resnet18(num_classes=10).to(device)

# torch.compile: traces and compiles the model to optimized CUDA kernels.
# First batch takes longer (compilation), but subsequent batches are faster.
# Gains are typically 10–30% on CNN workloads.
model = torch.compile(model)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

# CosineAnnealingLR: smoothly decays the learning rate from lr to near-zero
# over T_max epochs. Works better than step decay for most modern architectures.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scaler    = GradScaler()

# ── Training and validation loop ──────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds   = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total

for epoch in range(100):
    model.train()
    epoch_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), \
                         labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    scheduler.step()
    val_acc = evaluate(model, val_loader, device)
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1:>3} | loss: {avg_loss:.4f} | val_acc: {val_acc:.4f}")
```

> 💡 **Pro Tip:** `label_smoothing=0.1` in the loss function is a small regularization trick that prevents the model from becoming overconfident on training labels. It consistently improves generalization by 0.5–1% on classification tasks without any extra compute cost.

---

## 5.2 Transformer Training on a Single GPU

Transformers have different memory characteristics than CNNs. The self-attention mechanism scales quadratically with sequence length — doubling the sequence length quadruples the memory and compute for the attention layers. On a single GPU, this means long-sequence training hits VRAM limits quickly.

The standard toolkit for managing this: gradient checkpointing (to trade compute for memory), Flash Attention (a fused kernel that computes attention in blocks, dramatically reducing memory and improving speed), and mixed precision in BF16.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

device = torch.device("cuda")

# ── Load model with memory-saving options ─────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    # Flash Attention 2 requires pip install flash-attn
    # It replaces the default attention kernel with a tiled, memory-efficient version
    # attn_implementation="flash_attention_2",  # uncomment if flash-attn is installed
    torch_dtype=torch.bfloat16,                 # load weights in BF16
).to(device)

# Gradient checkpointing: stores ~1/sqrt(layers) activations instead of all of them.
# Reduces activation memory by 60-70% at the cost of ~30% more compute.
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ── Optimizer and scheduler ───────────────────────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(), lr=2e-5, weight_decay=0.01)

# Warmup + linear decay is standard for fine-tuning pre-trained models.
# Cold-starting at the full learning rate destabilises the pre-trained weights.
total_steps   = len(train_loader) * num_epochs
warmup_steps  = int(0.1 * total_steps)
scheduler     = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps,
    num_training_steps=total_steps)

scaler = GradScaler()

# ── Training step ─────────────────────────────────────────────────────
def train_step(batch):
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels         = batch["labels"].to(device)

    optimizer.zero_grad()
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    scaler.scale(outputs.loss).backward()

    # Gradient clipping prevents large gradient updates from destabilising
    # pre-trained weights during the early steps of fine-tuning.
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    return outputs.loss.item()
```

### Flash Attention — Why It Matters

Standard self-attention requires storing an N×N attention matrix in VRAM (N = sequence length). For N=2048 in FP16, a single layer's attention matrix costs over 16 MB — and this compounds across all layers and all batch elements.

Flash Attention computes attention in small tiles, reusing values from fast on-chip SRAM instead of writing intermediate results back to VRAM. The output is identical to standard attention, but VRAM usage grows linearly with sequence length instead of quadratically.

```bash
# Install Flash Attention (requires CUDA 11.6+ and Turing/Ampere GPU)
pip install flash-attn --no-build-isolation
```

---

## 5.3 Handling Datasets That Do Not Fit in RAM

When your dataset is too large to load into RAM, you need a streaming or memory-mapped approach. Both avoid loading the full dataset upfront — data is read from disk exactly when the model needs it.

### Memory-Mapped Arrays

NumPy's `memmap` maps a file on disk to an array-like object in Python. The OS handles paging — only the slices you actually access are loaded into RAM.

```python
import numpy as np
import torch
from torch.utils.data import Dataset

class MemmapDataset(Dataset):
    """
    Load large datasets without loading them into RAM.
    The file is memory-mapped — only accessed slices are paged in.
    """
    def __init__(self, features_path, labels_path, num_samples, feature_dim):
        # open='r' means read-only; the file is not loaded into RAM
        self.features = np.memmap(
            features_path, dtype='float32', mode='r',
            shape=(num_samples, feature_dim))
        self.labels   = np.memmap(
            labels_path, dtype='int64', mode='r',
            shape=(num_samples,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # .copy() is critical: memmap slices share memory with the file
        # buffer. Without .copy(), PyTorch's DataLoader workers can cause
        # race conditions when multiple workers access the same array.
        x = torch.from_numpy(self.features[idx].copy())
        y = int(self.labels[idx])
        return x, y

dataset = MemmapDataset("features.dat", "labels.dat",
                         num_samples=10_000_000, feature_dim=768)
loader  = DataLoader(dataset, batch_size=256, num_workers=8, pin_memory=True)
```

### Hugging Face Streaming Datasets

For text data, the Hugging Face `datasets` library supports streaming — iterating over a dataset hosted remotely without downloading the full file:

```python
from datasets import load_dataset

# streaming=True: yields examples one at a time, no local download
dataset = load_dataset("wikipedia", "20220301.en", streaming=True, split="train")

# Preprocess and batch on-the-fly
def tokenize(example):
    return tokenizer(
        example["text"], truncation=True, max_length=512, padding="max_length")

dataset = dataset.map(tokenize, batched=True, batch_size=1000)

# Works with standard PyTorch iteration
for batch in dataset.iter(batch_size=32):
    input_ids = torch.tensor(batch["input_ids"]).to(device)
    # ... run model
```

---

## 5.4 Multi-GPU Training Basics

When a single GPU is not enough — either because the model does not fit in one GPU's VRAM, or because you want to train faster by splitting work across multiple cards — PyTorch provides two main strategies.

### DataParallel — Quick but Limited

`nn.DataParallel` is a single-line wrapper that splits each batch across all available GPUs, runs the forward pass in parallel, then collects gradients back to GPU 0. It is easy to use but has real limitations: the gradient collection step creates a bottleneck on GPU 0, memory usage is uneven (GPU 0 carries more load), and it does not scale to multiple machines.

```python
import torch.nn as nn

model = YourModel()

if torch.cuda.device_count() > 1:
    print(f"Wrapping model for {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)   # splits batches automatically

model = model.cuda()
# Training loop is identical — DataParallel handles splitting and gathering
output = model(input_batch)   # batch split, forward run, results gathered
```

### DistributedDataParallel — The Right Choice

DistributedDataParallel (DDP) is more work to set up but significantly better in every practical way. Each GPU runs in its own process, owns a complete copy of the model, processes its own shard of data, and communicates gradients with other GPUs via NCCL (a high-performance GPU communication library). The gradient communication is overlapped with computation, so communication cost is largely hidden.

```python
# train_ddp.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup():
    """Initialise the process group. torchrun sets the env vars automatically."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def main():
    local_rank = setup()
    device     = torch.device(f"cuda:{local_rank}")

    # Each process creates its own model copy and wraps it in DDP
    model     = YourModel().to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    # DistributedSampler ensures non-overlapping data shards across processes
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank())
    loader  = DataLoader(
        train_dataset, batch_size=64, sampler=sampler,
        num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # set_epoch shuffles data differently each epoch across all processes
        sampler.set_epoch(epoch)

        ddp_model.train()
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = ddp_model(X)
            loss   = criterion(output, y)
            loss.backward()   # DDP automatically averages gradients across GPUs
            optimizer.step()

    cleanup()

if __name__ == "__main__":
    main()
```

Launch DDP training with `torchrun`, which handles spawning one process per GPU and setting the necessary environment variables:

```bash
# Single machine, 2 GPUs
torchrun --nproc_per_node=2 train_ddp.py

# Single machine, 4 GPUs
torchrun --nproc_per_node=4 train_ddp.py

# 2 machines, 4 GPUs each (8 total) — run on each machine
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=192.168.1.10 --master_port=12355 train_ddp.py
```

| | DataParallel | DistributedDataParallel |
|---|---|---|
| Setup complexity | 1 line | ~30 lines |
| Communication | GPU→CPU→GPU (slow) | GPU→GPU via NCCL (fast) |
| Memory balance | Uneven; GPU 0 overloaded | Even across all GPUs |
| Multi-machine support | No | Yes |
| When to use | Quick prototypes only | All real training jobs |

> ⚠️ **Common Mistake:** Using `DataParallel` for anything you care about. It looks simpler, and it is — but the GPU 0 memory imbalance will cause OOM errors for large models, and the CPU gather step can negate the benefit of multiple GPUs on small batches. Set up DDP from the start.

---

## Chapter Summary

CNN training on GPU is the most well-understood ML workflow — use it as a template. The full pipeline is: augmented DataLoader, ResNet-style model, AMP with GradScaler, cosine LR schedule, and `torch.compile` for a free extra speedup.

Transformer training adds two important differences: gradient checkpointing is almost always necessary to manage activation memory, and Flash Attention is worth installing if your work involves sequences longer than 512 tokens.

For datasets that exceed RAM, memmap handles structured numerical data and Hugging Face streaming handles text. Both integrate seamlessly with standard DataLoaders. For multi-GPU training, go directly to DDP — it is more setup but pays off immediately in memory efficiency and training speed.

---

*Next: Chapter 6 — GPU in the LLM Era. Why large language models push GPU limits in entirely new ways, and the practical toolkit for working with them: quantization, LoRA fine-tuning, and inference optimization.*
