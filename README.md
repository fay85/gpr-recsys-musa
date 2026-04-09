# GPR: Generative Pre-trained Recommender (MUSA)

A PyTorch implementation of the GPR model from ["GPR: Towards a Generative Pre-trained One-Model Paradigm for Large-Scale Advertising Recommendation"](https://arxiv.org/abs/2511.10138).

GPR replaces the traditional multi-stage cascading pipeline (retrieval → pre-ranking → ranking) with a single end-to-end generative model for advertising recommendation.

This version is adapted to run on **MUSA** (Moore Threads GPU) via `torch_musa`, with multi-GPU **FSDP** support mirroring the CUDA sibling repo `gpr-recsys` for accuracy alignment.

## Architecture

```
User Sequence → [HSD] → Intent Embeddings → [PTD] → Semantic IDs → [HTE] → Values
                  ↑                            ↑                       ↑
          Hybrid Attention            Thinking-Refining-         Per-level
          Token-Aware FFN              Generation                value heads
          Mixture-of-Recursions
```

### Key Components

| Component | Description |
|-----------|-------------|
| **RQ-KMeans+** | Residual quantizer that maps item embeddings to hierarchical semantic IDs. Combines K-means initialization with VAE-style fine-tuning. |
| **HSD** | Heterogeneous Sequence-wise Decoder with hybrid attention (bidirectional for prompt tokens, causal for items) and per-token-type FFN/LayerNorm. |
| **PTD** | Progressive Token-wise Decoder with learnable thinking tokens, diffusion-based refining, and autoregressive code generation. |
| **HTE** | Hierarchical Token-wise Evaluator that predicts business value at each semantic code level. |

### Training Pipeline

| Stage | Method | Purpose |
|-------|--------|---------|
| 1 | Multi-Token Prediction (MTP) | Pre-train multi-interest user modeling |
| 2 | Value-Aware Fine-Tuning (VAFT) | Align with business value (eCPM) |
| 3 | HEPO (RL) | Policy optimization with hierarchical process rewards |

## Setup

```bash
pip install -r requirements.txt
```

Ensure `torch_musa` is installed and a MUSA-capable GPU is available.

## Start Training

### Multi-GPU FSDP (recommended, matches CUDA version)

```bash
torchrun --nproc_per_node=8 train.py --dataset amazon
```

### Single-GPU fallback

```bash
python train.py --dataset amazon --no_fsdp
```

### Synthetic data (no download needed)

```bash
torchrun --nproc_per_node=8 train.py --dataset synthetic
```

### Run individual stages

```bash
# Stage 1: MTP pre-training
torchrun --nproc_per_node=8 train.py --stage mtp --dataset amazon

# Stage 2: VAFT (loads MTP checkpoint automatically)
torchrun --nproc_per_node=8 train.py --stage vaft --dataset amazon

# Stage 3: HEPO (loads VAFT checkpoint automatically)
torchrun --nproc_per_node=8 train.py --stage hepo --dataset amazon
```

### Resume from checkpoint

```bash
torchrun --nproc_per_node=8 train.py --stage vaft --resume checkpoints/mtp_best.pt
```

### Override batch size or epochs

```bash
torchrun --nproc_per_node=8 train.py --dataset amazon --batch_size 32 --mtp_epochs 10
```

### Launch TensorBoard

```bash
tensorboard --logdir runs/ --port 6006
```

### Evaluate

```bash
python evaluate.py --checkpoint checkpoints/gpr_final.pt --dataset synthetic
```

> **CUDA version:** The sibling repo `gpr-recsys` uses the identical model and training logic,
> with `nccl` backend instead of `mccl`:
> ```bash
> torchrun --nproc_per_node=8 train.py --dataset amazon
> ```

## MUSA-Specific Workarounds

This codebase includes workarounds for a known MuDNN Flash SDPA kernel crash when `seq_len=1`:

- **RefiningModule** (`model.py`): Manually unrolls `TransformerEncoderLayer` with `need_weights=True` to bypass SDPA.
- **PTD decoder** (`model.py`): `_decoder_layer_forward()` unrolls `TransformerDecoderLayer` with `need_weights=True`.

These workarounds force the `bmm` attention path instead of the Flash SDPA kernel.

## Configuration

All hyperparameters are in `config.py`. Key settings:

```python
# Model size (matches gpr-recsys CUDA version)
d_model = 1024         # hidden dimension
n_heads = 16           # attention heads
d_ff = 4096            # feed-forward dimension
n_layers_hsd = 24      # HSD transformer layers
n_layers_ptd = 6       # PTD decoder layers
n_semantic_levels = 3  # hierarchy depth
codebook_size = 2048   # codes per level
max_seq_len = 500      # sequence length limit

# Training
batch_size = 64        # per GPU
mtp_epochs = 30        # Stage 1
vaft_epochs = 15       # Stage 2
hepo_epochs = 10       # Stage 3
dtype = "bfloat16"     # mixed precision
```

## Data Schema

GPR uses four token types to represent the user journey:

| Token | Content | Example |
|-------|---------|---------|
| **U-Token** | User attributes | Demographics, preferences |
| **O-Token** | Organic content | Videos, articles browsed |
| **E-Token** | Environment/context | Time, device, position |
| **I-Token** | Ad items | Ads user interacted with |

Items are encoded as L-level semantic IDs via RQ-KMeans+ quantization.

## Project Structure

```
gpr-recsys-musa/
├── config.py           # Dataclass configurations
├── data_utils.py       # Data loading (Amazon Reviews + synthetic)
├── rq_tokenizer.py     # RQ-KMeans+ semantic ID tokenizer
├── model.py            # GPR model (HSD, PTD, HTE) + MUSA SDPA workarounds
├── train.py            # 3-stage training pipeline (FSDP multi-GPU)
├── evaluate.py         # Evaluation metrics
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@article{zhang2025gpr,
  title={GPR: Towards a Generative Pre-trained One-Model Paradigm for Large-Scale Advertising Recommendation},
  author={Zhang, Jun and Li, Yi and Liu, Yue and Wang, Changping and others},
  journal={arXiv preprint arXiv:2511.10138},
  year={2025}
}
```
