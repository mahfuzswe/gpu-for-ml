# Chapter 6 — GPU in the LLM Era

> *Large language models changed what it means to work with GPUs. Models that used to fit comfortably in 8 GB of VRAM now require multiple 80 GB cards — unless you know how to work around it. This chapter covers the practical toolkit for running and fine-tuning LLMs on real hardware.*

---

## 6.1 Why LLMs Are So GPU-Hungry

A 7B parameter language model sounds manageable until you do the math. At FP16 (2 bytes per parameter), the weights alone occupy 14 GB — already more than most consumer GPUs have available. And that is before you account for gradients, optimizer states, activations, and the KV cache.

The KV cache is a structure that is unique to autoregressive inference and has no equivalent in CNN or standard Transformer training. Every time the model generates a new token, it needs the key and value vectors from every previous token across every attention head and every layer. These vectors are cached to avoid recomputation, but the cache grows with every generated token. For a model with 32 layers, 32 attention heads, a head dimension of 128, and a context length of 4096 tokens, the KV cache in FP16 consumes roughly 8 GB on its own.

```
VRAM breakdown for a 7B model during inference (FP16, 4K context):

┌─────────────────────────────────────────────┐
│  Weights:        ~14.0 GB                   │
│  KV Cache:        ~4.0 GB  (grows with ctx) │
│  Activations:     ~1.5 GB  (per batch)      │
│  Overhead:        ~0.5 GB                   │
│  ─────────────────────────────────────────  │
│  Total:          ~20.0 GB                   │
│                                             │
│  → Minimum GPU: RTX 3090 / 4090 (24 GB)    │
└─────────────────────────────────────────────┘
```

During training, the situation is worse. Full fine-tuning of a 7B model in FP16 requires roughly 14 GB for weights, 14 GB for gradients, and 28 GB for Adam optimizer states (which keep a running mean and variance for every parameter) — totalling around 56 GB before activations. That requires at least an A100 80GB, or creative workarounds.

Those workarounds — quantization and LoRA — are what make LLM work accessible on consumer hardware, and they are what this chapter is about.

---

## 6.2 VRAM Math — Estimate Before You Load

Running out of VRAM halfway through loading a model wastes time. Get in the habit of estimating memory requirements before starting.

```python
def estimate_inference_vram_gb(
    param_billions: float,
    dtype: str = "fp16",
    context_length: int = 2048,
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    batch_size: int = 1
) -> dict:
    """
    Estimate VRAM requirements for LLM inference.
    Returns a breakdown of each component in GB.
    """
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
    bpp = bytes_per_param[dtype]

    # Model weights
    weights_gb = param_billions * 1e9 * bpp / 1e9

    # KV cache: 2 (K and V) × layers × heads × head_dim × context × batch × bytes
    kv_bytes    = 2 * num_layers * num_heads * head_dim * context_length * batch_size * bpp
    kv_cache_gb = kv_bytes / 1e9

    # Activations and buffers (rough estimate)
    overhead_gb = weights_gb * 0.1

    total_gb = weights_gb + kv_cache_gb + overhead_gb

    return {
        "weights_gb":   round(weights_gb,   2),
        "kv_cache_gb":  round(kv_cache_gb,  2),
        "overhead_gb":  round(overhead_gb,  2),
        "total_gb":     round(total_gb,     2),
    }

# Examples
print(estimate_inference_vram_gb(7,  "fp16"))   # FP16 LLaMA-7B
print(estimate_inference_vram_gb(7,  "int8"))   # INT8 LLaMA-7B
print(estimate_inference_vram_gb(7,  "int4"))   # INT4 / NF4 LLaMA-7B
print(estimate_inference_vram_gb(70, "fp16"))   # FP16 LLaMA-70B
```

The output gives you a concrete number to compare against your available VRAM before committing to a load. For full fine-tuning, multiply the weights estimate by roughly 4 to account for gradients and Adam states.

---

## 6.3 Quantization — Running Large Models on Small GPUs

Quantization reduces the numeric precision of model weights from FP16 (2 bytes) or FP32 (4 bytes) down to INT8 (1 byte) or INT4 (0.5 bytes). The model becomes smaller and loads faster, at some cost to output quality. For most inference tasks, the quality degradation from INT8 and even INT4 is small enough to be acceptable.

### Loading in 8-bit with bitsandbytes

The `bitsandbytes` library integrates with Hugging Face Transformers to enable transparent 8-bit loading. Linear layers in the model are stored in INT8; computations happen in FP16.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Llama-2-7b-hf"  # or any HF model

# load_in_8bit=True: weights stored as INT8, reducing VRAM from ~14 GB to ~8 GB
# device_map="auto": automatically distributes layers across available GPUs/CPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

inputs  = tokenizer("Explain gradient descent simply.", return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Loading in 4-bit with NF4 (QLoRA Standard)

4-bit NF4 (Normal Float 4) is the format used in QLoRA — the technique that made fine-tuning 7B models on a single RTX 3090 practical. NF4 is specifically designed so that the 16 discrete values it can represent are optimally spaced for neural network weights, which tend to follow a roughly normal distribution.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4: best quality for LLM weights
    bnb_4bit_use_double_quant=True,      # quantise the quantisation constants too
    bnb_4bit_compute_dtype=torch.bfloat16  # actual compute still in BF16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
# A 7B model now loads in approximately 4–5 GB of VRAM
```

| Method | VRAM (7B model) | Quality Loss | When to Use |
|---|---|---|---|
| FP32 | ~28 GB | None (baseline) | Not used for inference |
| FP16 / BF16 | ~14 GB | Negligible | Default for capable GPUs |
| INT8 (LLM.int8) | ~8 GB | Minimal | RTX 3070/3080 inference |
| NF4 (4-bit) | ~4–5 GB | Small | Consumer GPU inference and QLoRA |

> 💡 **Pro Tip:** For inference where quality matters, prefer INT8 over INT4. For fine-tuning with LoRA on limited VRAM, INT4 with NF4 is the standard approach and the quality tradeoff is small in practice.

---

## 6.4 LoRA and PEFT — Fine-Tuning Without Full VRAM

LoRA (Low-Rank Adaptation) is probably the most practically useful technique to come out of the LLM era for people without access to large GPU clusters. The core idea is simple: instead of updating all parameters during fine-tuning, freeze the original weights and inject small trainable matrices alongside selected layers. Only these small matrices receive gradient updates.

A 7B model has billions of parameters. LoRA might add a few million trainable parameters — less than 0.1% of the total — and fine-tune only those. The frozen base model requires no gradients and no optimizer states, which eliminates most of the memory overhead of fine-tuning. Combined with 4-bit quantization (this combination is called QLoRA), a 7B model can be fine-tuned on a single RTX 3090 in a few hours.

```
                    Standard Fine-tuning
  Weight Matrix W  →  W + ΔW   (full ΔW: same size as W)

                    LoRA Fine-tuning
  Weight Matrix W  →  W + B·A  (frozen W, tiny trainable B and A)

  Where: B is (d × r), A is (r × k), rank r << d, r << k
  Memory for ΔW:   d × k parameters
  Memory for B·A:  (d+k) × r parameters   (e.g. 100× smaller when r=8)
```

### Setting Up QLoRA Fine-Tuning

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch

model_id = "meta-llama/Llama-2-7b-hf"

# Step 1: Load model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto")

# Step 2: Prepare for k-bit training
# This casts layer norms to FP32 and enables gradient checkpointing
# in a way that is compatible with quantized weights.
model = prepare_model_for_kbit_training(model)

# Step 3: Configure LoRA
# target_modules: which linear layers to add LoRA adapters to.
# For LLaMA-style models, q_proj and v_proj (query and value projections
# in attention) are the standard targets.
# r: the rank. Higher rank = more capacity = more trainable parameters.
# lora_alpha: scaling factor. A common rule: set to 2× the rank.
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)

# See how few parameters are actually trainable
model.print_trainable_parameters()
# Example output:
# trainable params: 20,971,520 || all params: 6,758,367,232
# trainable%: 0.3102%
```

### Training the Adapters

```python
from trl import SFTTrainer
from transformers import TrainingArguments

# Prepare a simple dataset
# Each example should be a dict with a "text" field (or whatever field_name you set)
# Example: {"text": "### Instruction: ...\n### Response: ..."}

training_args = TrainingArguments(
    output_dir="./qlora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # effective batch size = 16
    learning_rate=2e-4,
    fp16=True,                         # compute in FP16
    bf16=False,
    logging_steps=25,
    save_strategy="epoch",
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to="none",                  # set to "wandb" to enable experiment tracking
    optim="paged_adamw_8bit",          # 8-bit paged optimizer: saves VRAM on optimizer states
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args,
)

trainer.train()

# Save only the LoRA adapter weights (much smaller than the full model)
model.save_pretrained("./qlora-adapters")
```

### Merging Adapters Back Into the Base Model

After fine-tuning, you can either keep the adapters separate (load base model + adapters at inference time) or merge them into the base model for a standalone checkpoint:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model in FP16 for merging
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto")

# Load LoRA adapters on top
peft_model = PeftModel.from_pretrained(base_model, "./qlora-adapters")

# Merge LoRA weights into base model — result is a standard HF model
merged_model = peft_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("./merged-model")
```

---

## 6.5 Inference Optimization

Fine-tuning produces a good model. Running it efficiently in production requires a few more considerations.

### vLLM — High-Throughput Serving

Naive Hugging Face `model.generate()` processes one request at a time and does not efficiently batch concurrent requests. For serving a model to multiple users or processing large batches of inputs, [vLLM](https://github.com/vllm-project/vllm) is the current standard.

vLLM uses **PagedAttention** — a memory management scheme that treats the KV cache like virtual memory, allocating it in fixed-size pages rather than contiguously. This prevents memory fragmentation and allows far more requests to be batched simultaneously.

```bash
pip install vllm

# Serve a model with an OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85
```

```python
# Offline batch inference with vLLM
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    dtype="bfloat16",
    gpu_memory_utilization=0.85,  # reserve 15% for KV cache headroom
)

prompts = [
    "Explain backpropagation to a 10-year-old.",
    "What is the difference between FP16 and BF16?",
    "Write a Python function that checks if a number is prime.",
]
params  = SamplingParams(temperature=0.7, max_tokens=300)

outputs = llm.generate(prompts, params)
for out in outputs:
    print(f"Prompt: {out.prompt[:50]}...")
    print(f"Response: {out.outputs[0].text}\n")
```

### Key Inference Optimizations at a Glance

| Technique | What It Does | How to Enable |
|---|---|---|
| KV cache | Caches past attention K/V to avoid recomputation | Enabled by default in HF `generate()` |
| BF16 inference | Halves weight memory vs FP32 | `torch_dtype=torch.bfloat16` |
| `torch.compile` | Fuses kernels for faster forward pass | `model = torch.compile(model)` |
| Flash Attention 2 | Reduces attention memory from O(n²) to O(n) | `attn_implementation="flash_attention_2"` |
| vLLM + PagedAttention | Batches concurrent requests efficiently | Run vLLM server |
| Quantized inference | INT4/INT8 weights for smaller memory footprint | `load_in_4bit=True` via bitsandbytes |

---

## Chapter Summary

The LLM era changed GPU workflows fundamentally. Model size now routinely exceeds consumer VRAM in FP16, so quantization is not an optimisation — it is the default starting point. INT8 loading cuts VRAM roughly in half with minimal quality loss. NF4 (4-bit) cuts it by another 2×, enabling 7B models to fit on a 6 GB GPU.

LoRA makes fine-tuning accessible by making nearly all parameters frozen and training only a small set of injected low-rank matrices. Combined with 4-bit quantization (QLoRA), you can fine-tune a 7B model on a single RTX 3090 in a few hours. The resulting adapters are a few hundred megabytes — a fraction of the full model size.

For inference at scale, vLLM's PagedAttention is the practical standard. For offline inference or personal use, direct `model.generate()` with BF16 and Flash Attention works well.

---

*Next: Chapter 7 — Cloud and Remote GPU Usage. Using Colab, Kaggle, and paid cloud services effectively, plus remote machine workflows with SSH.*
