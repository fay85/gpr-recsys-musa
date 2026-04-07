# GPR: Generative Pre-trained Recommender (MUSA)

A PyTorch implementation of the GPR model from ["GPR: Towards a Generative Pre-trained One-Model Paradigm for Large-Scale Advertising Recommendation"](https://arxiv.org/abs/2511.10138).

GPR replaces the traditional multi-stage cascading pipeline (retrieval → pre-ranking → ranking) with a single end-to-end generative model for advertising recommendation.

This version is adapted to run on **MUSA** (Moore Threads GPU) via `torch_musa`.

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

## Quick Start

### Train with synthetic data (no download needed)

```bash
python train.py --dataset synthetic --batch_size 128
```

### Train with Amazon Reviews

```bash
python train.py --dataset amazon --batch_size 128
```

The Amazon Beauty dataset will be downloaded automatically.

### Run individual stages

```bash
# Stage 1: MTP pre-training
python train.py --stage mtp --dataset synthetic

# Stage 2: VAFT (loads MTP checkpoint automatically)
python train.py --stage vaft

# Stage 3: HEPO (loads VAFT checkpoint automatically)
python train.py --stage hepo
```

### Evaluate

```bash
python evaluate.py --checkpoint checkpoints/gpr_final.pt --dataset synthetic
```

## Configuration

All hyperparameters are in `config.py`. Key settings:

```python
# Model size
d_model = 128          # hidden dimension
n_heads = 4            # attention heads
n_layers_hsd = 4       # HSD transformer layers
n_layers_ptd = 2       # PTD decoder layers
n_semantic_levels = 3  # hierarchy depth
codebook_size = 256    # codes per level

# Training
mtp_epochs = 30        # Stage 1
vaft_epochs = 15       # Stage 2
hepo_epochs = 10       # Stage 3
batch_size = 128
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
├── model.py            # GPR model (HSD, PTD, HTE)
├── train.py            # 3-stage training pipeline
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
