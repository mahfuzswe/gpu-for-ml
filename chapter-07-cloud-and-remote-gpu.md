# Chapter 7 — Cloud and Remote GPU Usage

> *Not everyone has a high-end GPU at their desk. Cloud and hosted platforms close that gap — if you know how to use them without wasting compute time or running into avoidable limitations.*

---

## 7.1 Understanding the Trade-offs Before You Start

Before diving into specific platforms, it helps to think clearly about what you are actually trading when you use cloud GPUs versus local hardware.

With a **local GPU**, you pay once (the hardware cost), then have unlimited access with no per-hour anxiety. You can run a 3-hour debugging session that accomplishes nothing and it costs you nothing extra. The downside is that VRAM is fixed — if your model needs 40 GB and your card has 24 GB, no amount of optimization changes that ceiling.

With **cloud GPUs**, you get access to hardware you could not otherwise afford — A100s, H100s, multi-GPU setups — but every idle minute costs money. This changes how you work: you tend to write and debug code locally (or on a free tier), then spin up the expensive hardware only when the code is ready to run.

The practical takeaway is that the two approaches are complementary, not competing. Use free tiers and local hardware for development and debugging. Use cloud for jobs that genuinely need more VRAM or multiple GPUs.

---

## 7.2 Google Colab

Google Colab is the most accessible GPU platform for beginners. It runs Python notebooks in a browser, connects to Google Drive, and provides free GPU access — usually a T4 (16 GB VRAM), occasionally an A100 on the free tier, and more reliably on paid tiers.

### Getting the Most From the Free Tier

The free tier has real limitations: sessions disconnect after roughly 12 hours (and sooner if the browser tab is idle), GPU allocation is not guaranteed, and the available GPU type changes based on demand. Working around these constraints is mostly about habits:

```python
# ── Always check what GPU you actually got ────────────────────────────
import subprocess
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout)
# The GPU type significantly affects what you can run.
# T4: good for inference and small models.
# A100: fine-tune 7B+ models comfortably.

# ── Mount Google Drive immediately — your files need to survive ────────
from google.colab import drive
drive.mount("/content/drive")

# Save everything important to Drive, not /content/ (which is lost on disconnect)
CHECKPOINT_DIR = "/content/drive/MyDrive/ml_checkpoints/"
import os; os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Install dependencies at the top of every notebook ─────────────────
# Colab sessions reset between disconnections — pip installs do not persist.
# Keep all installs in one cell so re-running is fast.
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "transformers", "peft", "trl", "bitsandbytes", "datasets", "accelerate"
])
```

Saving checkpoints frequently is not optional on Colab — it is essential. A session can end with no warning and the entire `/content/` directory disappears. Save to Drive every epoch, or at minimum every 30 minutes of training:

```python
import torch

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save a full training checkpoint to Google Drive."""
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss":                 loss,
    }, path)
    print(f"  Checkpoint saved → {path}")

def load_checkpoint(model, optimizer, path):
    """Resume from a checkpoint if it exists."""
    if not os.path.exists(path):
        print("No checkpoint found — starting from scratch.")
        return 0, float("inf")

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  Resumed from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"], checkpoint["loss"]

# In your training loop:
for epoch in range(start_epoch, num_epochs):
    # ... training code ...
    save_checkpoint(
        model, optimizer, epoch, avg_loss,
        f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch}.pt"
    )
```

### Colab Pro and Pro+

Colab Pro (\$9.99/month) gives priority GPU access, longer sessions, and more reliable A100 availability. Colab Pro+ (\$49.99/month) extends this further with background execution (your notebook runs even with the browser closed) and even longer sessions.

For occasional heavy training runs, Pro is worth the cost. For daily ML work, it is cheaper to buy an RTX 4070 or rent from RunPod.

> 💡 **Pro Tip:** On the free tier, connect to the runtime and immediately run a cell — then keep the browser tab active. Colab measures inactivity from the last cell execution, not from when you opened the tab. An extension like `Auto-Reconnect` in the browser can help with the connection side.

---

## 7.3 Kaggle Notebooks

Kaggle provides 30 free GPU-hours per week, with reliable access to T4 or P100 GPUs (both 16 GB VRAM). Unlike Colab, Kaggle sessions run for up to 12 hours regardless of browser activity, and output files persist automatically in `/kaggle/working/`.

The main practical difference from Colab: Kaggle is built around competitions and datasets, so it handles large public datasets natively. You can attach any Kaggle dataset to your notebook and access it without downloading.

```python
# Inside a Kaggle notebook — paths are fixed
INPUT_DIR  = "/kaggle/input/"   # datasets attached to this notebook (read-only)
OUTPUT_DIR = "/kaggle/working/" # your files — persists across sessions

# List attached datasets
import os
for d in os.listdir(INPUT_DIR):
    print(d)

# Save your model output
import torch
torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_weights.pt")
# This file persists and can be downloaded from the notebook's Output tab
```

Kaggle is particularly well-suited for:
- Competition workflows where public datasets are already hosted on the platform
- Experiments that fit within 12-hour sessions without needing to manually keep a browser open
- Situations where you need 30 hours per week consistently but do not want to pay

The main limitation is that the 30 hours reset weekly and are not rollable — unused hours do not carry over. If you need more than 30 hours in a week, you are paying.

---

## 7.4 Paid GPU Cloud Services

When free tiers are not enough, several services offer on-demand GPU access at rates that are significantly cheaper than AWS or GCP for pure compute workloads.

| Service | Notable GPUs | Pricing (approx.) | Best For |
|---|---|---|---|
| [Lambda Labs](https://lambdalabs.com) | A100, H100, A10 | \$1.10–\$3.50/hr | Research, stable on-demand |
| [RunPod](https://runpod.io) | RTX 4090, A100, H100 | \$0.74–\$4.69/hr | Flexible spot + on-demand |
| [Vast.ai](https://vast.ai) | RTX 3090, A100, H100 | \$0.40–\$3.00/hr | Lowest rates (spot, peer-hosted) |
| [AWS (p4d, p5)](https://aws.amazon.com) | A100, H100 | \$12–\$32+/hr | Enterprise, compliance needs |
| [Google Cloud A3](https://cloud.google.com) | H100 | \$12–\$40+/hr | TPU access, GCP ecosystem |

For most ML researchers and independent engineers, Lambda Labs or RunPod hits the right balance of reliability, pricing, and ease of use. Vast.ai is the cheapest but the hardware is peer-hosted, so reliability varies by provider.

> 🚀 **Best Practice:** For long training runs (24+ hours), always use **spot or interruptible instances** and implement checkpoint-and-resume. Spot instances are typically 3–5× cheaper than on-demand. A run that saves a checkpoint every 30 minutes can be interrupted and resumed without losing significant progress.

---

## 7.5 SSH Into Remote GPU Machines

Whether you are using a cloud VM, a university cluster, or a colleague's machine, the SSH-based workflow is the same. The core tools are: SSH for connection, tmux for persistent sessions, rsync for code synchronisation, and SSH port forwarding for Jupyter notebooks.

### Persistent Sessions with tmux

Without tmux, disconnecting from SSH kills your running process. With tmux, your processes continue running inside a tmux session on the remote machine, and you can reattach to them from any connection:

```bash
# Connect to the remote machine
ssh user@gpu-server-ip

# Start a new named tmux session
tmux new-session -s training

# Run your training job inside tmux
python train.py --epochs 100

# Detach from the session (your job keeps running)
# Press: Ctrl+B, then D

# --- Later, reconnect ---
ssh user@gpu-server-ip
tmux attach-session -t training   # reconnect to the running session

# List all active sessions
tmux ls
```

### Syncing Code with rsync

Editing code locally and syncing it to the remote machine is faster than editing directly over SSH, especially for large projects:

```bash
# Push your local project directory to the remote machine
# --exclude flags prevent syncing .git history, caches, and output files
rsync -avz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='outputs/' \
    --exclude='.venv/' \
    ./my-project/ user@gpu-server:/home/user/my-project/

# Pull results and model checkpoints back to local machine
rsync -avz user@gpu-server:/home/user/my-project/outputs/ ./outputs/
```

### Jupyter Notebooks Over SSH Port Forwarding

Port forwarding lets you run a Jupyter notebook server on the remote GPU machine and access it through your local browser, as if it were running locally:

```bash
# Step 1: On the remote machine — start Jupyter without launching a browser
jupyter notebook --no-browser --port=8888

# Step 2: On your local machine — forward the remote port to your local machine
# -N: do not execute a remote command (just forward)
# -L: local port 8888 → remote port 8888 on gpu-server
ssh -N -L 8888:localhost:8888 user@gpu-server-ip

# Step 3: Open in your local browser
# http://localhost:8888
# Use the token shown in the jupyter output on the remote machine
```

For VS Code users, the Remote-SSH extension provides a cleaner workflow: you connect to the remote machine and edit files directly as if they were local, with the full VS Code experience.

### Monitoring a Remote Training Job

```bash
# Watch GPU usage on the remote machine from your local terminal
ssh user@gpu-server 'watch -n 2 nvidia-smi'

# Or tail the training log
ssh user@gpu-server 'tail -f ~/my-project/training.log'

# Send training output to a log file while keeping it printable
# Run this inside tmux on the remote machine
python train.py 2>&1 | tee training.log
```

---

## Chapter Summary

Colab and Kaggle cover most beginner and intermediate GPU needs for free. Colab's T4 handles anything up to medium-sized model training or inference; Kaggle's 30 free hours per week are reliable for competition work and experiments that fit within a 12-hour window. The most important habit for both is saving checkpoints aggressively — a session ending unexpectedly should never cost you more than 30 minutes of work.

When free tiers are not enough, RunPod and Lambda Labs offer the best cost-per-GPU-hour for research workloads. Use spot instances for long runs and implement checkpoint-and-resume to protect against interruption.

For any remote machine — cloud or institutional — the tmux, rsync, and SSH port forwarding combination covers nearly every workflow. Learn these three tools and remote GPU work becomes as comfortable as working locally.

---

*Next: Chapter 8 — Real-World Engineering Tips. Choosing the right GPU, decoding CUDA error messages, and the beginner mistakes that account for most debugging time.*
