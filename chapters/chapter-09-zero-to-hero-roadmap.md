# Chapter 9 — Zero to Hero Roadmap

> *Knowing the techniques is one thing. Knowing the order to learn them is another. This chapter gives you a structured path from your first GPU script to production-quality ML engineering, along with the tools that make the journey faster.*

---

## 9.1 How to Use This Roadmap

The ten-week path below is organised so that each week produces something tangible — code that runs, a model that trains, a technique that visibly improves performance. This matters because GPU programming has a feedback loop problem: the tools and concepts are interdependent, and learning them abstractly without running real code tends to produce knowledge that evaporates quickly.

Each week has a specific goal and a concrete deliverable. The weeks are sequential because the concepts stack — mixed precision assumes you understand the training loop, gradient accumulation assumes you understand mixed precision, and so on. If you already have a foundation in some of these areas, skip ahead, but read the goal statement first to make sure you are not missing a specific skill.

---

## 9.2 Week-by-Week Learning Path

### Weeks 1–2: Foundation

**Goal:** Get your GPU environment working and understand every line of a basic PyTorch training loop. By the end of week 2, you should be able to write a GPU training loop from memory without looking anything up.

The first week is about removing friction. Environment setup is not glamorous, but spending time here means you never lose momentum later to a CUDA version mismatch or a failed pip install. Follow Chapter 2 step by step. When `torch.cuda.is_available()` returns `True` and your GPU name appears in the verification script, you are done with setup.

Week 2 is about internalising the training loop pattern from Chapter 3. Write it by hand, not by copying from this guide. Train something simple — a two-layer network on MNIST or a CNN on CIFAR-10. Open `nvidia-smi` in a terminal beside your code and watch the GPU-Util and Memory-Usage change as training starts. This is how you develop intuition for what "the GPU is working" looks like versus "the GPU is idle."

**Deliverable:** A CNN that trains on CIFAR-10, monitored with `nvidia-smi`, that you can reproduce from a blank file.

---

### Weeks 3–4: Optimization Fundamentals

**Goal:** Make your training loop fast. Understand and apply the four core techniques from Chapter 4.

Start with mixed precision — it is the highest-impact change and the lowest risk. Add `autocast` and `GradScaler` to your CIFAR-10 loop, run it, and measure the speedup. The difference is usually visible immediately in `nvidia-smi`'s Power Usage row and in wall-clock time.

Then fix your DataLoader. Set `num_workers=4` and `pin_memory=True`, watch GPU-Util in `nvidia-smi`, and compare it to the default `num_workers=0`. If GPU-Util increases noticeably, your DataLoader was the bottleneck. Then add gradient accumulation to simulate a larger batch size than your VRAM allows alone.

Finally, run the PyTorch Profiler on your training loop. Look at the output and find the most expensive operation. This is the first time you will see what your code actually does at the GPU level, and it changes how you think about performance.

**Deliverable:** Your CIFAR-10 training loop running with mixed precision, an optimised DataLoader, and gradient accumulation — with profiler output showing the top 10 most expensive operations.

---

### Weeks 5–6: Transformers and Hugging Face

**Goal:** Fine-tune a pre-trained transformer on a real NLP task and understand the memory differences compared to CNN training.

Fine-tuning BERT for text classification is the right starting point. The Hugging Face `datasets` library provides dozens of standard benchmarks (SST-2 for sentiment, MNLI for inference, SQuAD for reading comprehension) that you can load in two lines. Fine-tune on one, use the Hugging Face `Trainer` for the first run, then replace it with a manual training loop to understand what the `Trainer` abstracts away.

The specific learning this week comes from memory. Enable gradient checkpointing and measure VRAM before and after. Load a sequence of increasing lengths and observe the quadratic memory growth of attention. If your GPU supports it, install `flash-attn` and measure the memory and speed difference. This makes the abstract "attention is O(n²)" statement concrete and memorable.

**Deliverable:** BERT fine-tuned on a text classification dataset, with gradient checkpointing enabled, and a measurement of VRAM usage at sequence lengths 128, 256, 512, and 1024.

---

### Weeks 7–8: LLMs and Quantization

**Goal:** Load and run a 7B language model. Fine-tune it on a custom dataset using QLoRA. Understand the memory arithmetic well enough to estimate requirements before loading.

The first step is loading a 7B model and running inference. Use the VRAM estimation function from Chapter 6 to predict the memory requirement before you load it, then compare the prediction to the actual `nvidia-smi` output after loading. This builds numerical intuition for LLM memory that serves every subsequent project.

Then fine-tune. Create a small custom dataset — instructions and responses in a domain you care about, even a few hundred examples. Run QLoRA fine-tuning using `SFTTrainer`. Compare outputs before and after fine-tuning on your domain-specific inputs. Save and reload the LoRA adapters. Merge the adapters into the base model and save the result.

**Deliverable:** A 7B model fine-tuned with QLoRA on a custom domain dataset, with before-and-after inference comparisons and a merged adapter checkpoint.

---

### Weeks 9–10: Cloud and Production Patterns

**Goal:** Run a real training job on cloud infrastructure. Implement checkpoint-and-resume. Set up an inference server.

Spin up a RunPod or Lambda Labs instance. SSH in using the workflow from Chapter 7 — tmux for persistence, rsync to push your code, SSH port forwarding to access Jupyter. Run the QLoRA training job you developed in weeks 7–8 on this cloud instance. Implement checkpoint-and-resume and deliberately interrupt the training partway through to verify that it recovers correctly.

If you have access to a machine with two GPUs (some cloud instances have this), set up a DDP training job with `torchrun`. Confirm that both GPUs show high utilisation in `nvidia-smi` and that the effective batch size doubles.

Finally, serve the fine-tuned model locally with vLLM. Send requests to it from a Python script and measure tokens per second at batch size 1, 4, and 16. This completes the loop from raw model weights to a working inference endpoint.

**Deliverable:** A checkpoint-resumable training job that has actually been interrupted and recovered, and a vLLM inference server with a measured throughput benchmark.

---

## 9.3 Recommended Tool Stack

The tools below are mainstream, well-documented, and chosen to complement each other rather than overlap. You do not need everything on this list immediately — add tools as the need arises.

| Category | Tool | Why This One |
|---|---|---|
| **Core framework** | PyTorch | Dominant in research; best debugging experience |
| **Transformers** | Hugging Face Transformers | Standard library for pre-trained models |
| **Fine-tuning** | PEFT + bitsandbytes | LoRA, QLoRA, and quantization in one ecosystem |
| **Training helpers** | Hugging Face TRL (SFTTrainer) | Handles LLM fine-tuning boilerplate |
| **Experiment tracking** | Weights & Biases (wandb) | Loss curves, GPU metrics, hyperparameter comparison |
| **GPU monitoring** | nvitop | Better than nvidia-smi for interactive use |
| **LLM serving** | vLLM | PagedAttention; the production standard |
| **Edge inference** | llama.cpp | CPU/quantized inference when no GPU is available |
| **Datasets** | Hugging Face Datasets | Streaming, caching, and tokenization included |
| **Environment** | conda or uv | Dependency isolation; conda for CUDA-heavy setups |

One note on experiment tracking: `wandb` is worth setting up early, even for personal projects. Losing track of which hyperparameters produced which results is a genuine problem after the fifth or sixth experiment. The free tier of wandb handles this with almost no setup cost:

```python
import wandb

wandb.init(project="gpu-mastery-experiments", config={
    "learning_rate": 3e-4,
    "batch_size":    128,
    "epochs":        50,
    "model":         "resnet18",
})

# Log metrics each epoch
for epoch in range(num_epochs):
    train_loss, val_acc = train_epoch(model, loader)
    wandb.log({"train_loss": train_loss, "val_acc": val_acc, "epoch": epoch})

wandb.finish()
```

---

## 9.4 Suggested Mini-Projects

Reading and following tutorials builds familiarity, but mini-projects build actual competence. The difference is that a project has an open-ended problem you have to solve yourself, not a known answer to reproduce. These are ordered roughly by difficulty, but any of them is appropriate after completing the relevant chapters.

**Project 1 — GPU Benchmark Tool** (after Chapter 3): Write a script that measures FP32 and FP16 matrix multiply throughput on your GPU across a range of matrix sizes (512, 1024, 2048, 4096, 8192). Plot the results. Compare the FP16/FP32 ratio — it tells you how well your GPU's Tensor Cores are being utilised. Try to match NVIDIA's published TFLOPS for your card.

**Project 2 — CIFAR-10 Speedrun** (after Chapter 4): Train ResNet-18 to 93% accuracy on CIFAR-10 in under 5 minutes of wall-clock time, using every optimization technique from Chapter 4. This requires careful profiling and tuning rather than just throwing techniques at the problem blindly.

**Project 3 — VRAM Profiler** (after Chapter 5): Build a tool that hooks into a transformer model's forward pass and measures VRAM usage after each layer. Visualise the memory profile as a bar chart. See which layers consume the most memory and how gradient checkpointing changes the profile.

**Project 4 — Domain Fine-Tune** (after Chapter 6): Fine-tune a 3B or 7B model on a domain you actually care about — cooking, medical literature, code in a specific language, historical texts. Build an evaluation script that tests before-and-after performance on 20 domain-specific prompts. Write up what changed and what did not.

**Project 5 — Inference API** (after Chapter 6–7): Serve a quantized 7B model with vLLM behind a FastAPI endpoint. Add streaming output. Write a benchmark client that sends 50 concurrent requests and measures latency distribution (p50, p95, p99) and throughput (tokens/second) at batch sizes 1, 4, 8, and 16.

**Project 6 — DDP Scaling Experiment** (after Chapter 5, with 2+ GPUs): Train ResNet-50 with DDP using 1 GPU, then 2 GPUs. Measure throughput (samples per second) for each configuration. The ideal result is near-linear scaling — 2 GPUs should be close to 2× the throughput. Identify and explain any gap between ideal and observed scaling.

---

## Chapter Summary

The ten-week path progresses through setup, optimization, transformer fine-tuning, LLMs, and cloud workflows — each week producing something you can actually run and inspect. The sequence matters because the concepts are interdependent, and the deliverables matter because passive reading does not produce GPU programming skill.

The recommended stack covers the full lifecycle from training to serving, is well-documented, and is standard across most ML engineering teams. Add tools as you need them, not before.

The mini-projects are where the deepest learning happens. Each one requires you to solve an open-ended problem rather than following a walkthrough, and that gap — knowing the technique versus being able to apply it independently — is exactly what they are designed to bridge.

---

## Navigation

**[← Previous: Chapter 8 — Real-World Engineering](chapter-08-real-world-engineering.md)**

**[➜ Next: Chapter 10 — Appendix](chapter-10-appendix.md)** — Command cheat sheet, error reference, and curated resources.
