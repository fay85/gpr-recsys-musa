"""
GPR Training Pipeline with FSDP + TensorBoard Monitoring.

Three-stage training:
  Stage 1: Pre-training with Multi-Token Prediction (MTP)
  Stage 2: Value-Aware Fine-Tuning (VAFT)
  Stage 3: Post-training with HEPO (Hierarchical Enhanced Policy Optimization)

Launch with torchrun for multi-GPU FSDP:
  torchrun --nproc_per_node=8 train.py --dataset amazon

Single-GPU fallback (no FSDP):
  python train.py --dataset amazon --no_fsdp

All losses, metrics, and learning rates are logged to TensorBoard.
"""

import os
import csv
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", message="Expected query, key and value to all be of dtype")

import argparse
import functools
import random
import time
import json
from datetime import datetime

import sys
import traceback

import numpy as np
import torch
import torch_musa

import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# MUSA SDPA workaround: replace F.scaled_dot_product_attention with a
# pure-math ("eager") implementation.  MuDNN Flash / MemEfficient SDPA
# kernels segfault on certain shapes, crashing the process with no Python
# traceback.  The eager version uses only matmul + softmax — no native
# SDPA kernels — so it is fully reliable on any backend.
# ---------------------------------------------------------------------------
def _eager_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None):
    L, S = query.size(-2), key.size(-2)
    scale_factor = scale if scale is not None else query.size(-1) ** -0.5
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if is_causal:
        causal_mask = torch.ones(
            L, S, dtype=torch.bool, device=query.device
        ).triu(diagonal=1)
        attn_weight = attn_weight.masked_fill(causal_mask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(attn_mask, float("-inf"))
        else:
            attn_weight = attn_weight + attn_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p > 0.0:
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
    return attn_weight @ value

if hasattr(F, "scaled_dot_product_attention"):
    F.scaled_dot_product_attention = _eager_sdpa
    print("[MUSA] SDPA: patched to eager (math-only) — Flash/MemEfficient disabled")

import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from tqdm import tqdm

from config import GPRConfig
from data_utils import (
    generate_synthetic_data,
    load_amazon_reviews,
    build_sequences,
    create_dataloaders,
    build_user_code_popularity,
    get_all_codes_per_level,
    generate_arr_samples,
    merge_batches,
)
from rq_tokenizer import RQKMeansPlus
from model import GPR, SemanticTrie, mtp_loss, vaft_loss, HSDBlock

_DBG_TRAIN = os.environ.get("GPR_DEBUG", "0") == "1"
_TRACE = os.environ.get("GPR_TRACE", "1") == "1"


def _trace(msg):
    """Checkpoint-style trace: sync MUSA, print step + GPU memory, flush.

    Controlled by GPR_TRACE env var (default ON for debugging).
    The torch.musa.synchronize() ensures all queued MUSA ops finish
    *before* the log line, so the last printed step is the true crash point.
    """
    if not _TRACE:
        return
    try:
        torch.musa.synchronize()
    except Exception:
        pass
    r = rank() if dist.is_initialized() else 0
    try:
        alloc = torch.musa.memory_allocated() / 1024**3
        resv = torch.musa.memory_reserved() / 1024**3
        mem = f"mem={alloc:.2f}/{resv:.2f}GB"
    except Exception:
        mem = "mem=?"
    print(f"[rank{r}][TRACE] {msg}  ({mem})", flush=True)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_dist():
    return dist.is_initialized()


def rank():
    return dist.get_rank() if is_dist() else 0


def world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    return rank() == 0


def local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def print0(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)


def setup_distributed():
    if "RANK" not in os.environ:
        return False
    lr = local_rank()
    torch.musa.set_device(lr)
    dist.init_process_group(backend="mccl", device_id=torch.device("musa", lr))
    return True


def cleanup_distributed():
    if is_dist():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------

def wrap_model_fsdp(model, cfg):
    """Wrap GPR model with FSDP using per-HSDBlock sharding."""
    device_id = local_rank()
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # NOTE: nn.TransformerDecoderLayer is intentionally excluded.
    # The MUSA SDPA workaround in PTD._decoder_layer_forward manually
    # unrolls decoder-layer internals (layer.self_attn, etc.).  If FSDP
    # individually wraps each DecoderLayer, those sub-module accesses
    # bypass FSDP's unshard hook and hit flattened parameters, causing
    # "Dimension out of range" errors.  PTD's 6 layers are small enough
    # that leaving them in the root FSDP unit has negligible memory cost.
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            HSDBlock,
        },
    )

    _trace("wrap_model_fsdp: before FSDP()")
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=bf16_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device_id,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    _trace("wrap_model_fsdp: FSDP() done")

    if cfg.train.activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
        )
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: isinstance(m, (HSDBlock,)),
        )
        _trace("wrap_model_fsdp: activation checkpointing applied")

    return model


# ---------------------------------------------------------------------------
# Checkpoint helpers for FSDP
# ---------------------------------------------------------------------------

def save_checkpoint(model, path, is_fsdp=False):
    if is_fsdp:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state = model.state_dict()
        if is_main():
            torch.save(state, path)
    else:
        torch.save(model.state_dict(), path)


def load_checkpoint(model, path, is_fsdp=False):
    if is_fsdp:
        full_sd = None
        if is_main():
            full_sd = torch.load(path, weights_only=True, map_location="cpu")
        if is_dist():
            objects = [full_sd]
            dist.broadcast_object_list(objects, src=0)
            full_sd = objects[0]
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model.load_state_dict(full_sd)
    else:
        model.load_state_dict(torch.load(path, weights_only=True))


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = True):
    """
    Seed all RNGs and optionally enforce deterministic algorithms.
    This ensures bit-identical results across runs and across backends
    (CUDA vs MUSA) for accuracy alignment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.musa.is_available():
        torch.musa.manual_seed(seed)
        torch.musa.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["MUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if hasattr(torch.backends, "mudnn"):
            torch.backends.mudnn.deterministic = True
            torch.backends.mudnn.benchmark = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def to_device(batch: dict, device, dtype: torch.dtype) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(dtype)
        out[k] = v
    return out


def clip_grad_norm(model, max_norm, is_fsdp=False):
    if is_fsdp:
        return model.clip_grad_norm_(max_norm)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# ---------------------------------------------------------------------------
# TensorBoard helper (rank-0 only)
# ---------------------------------------------------------------------------

class TBLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir) if is_main() else None
        self.global_step = 0

    def log_scalars(self, tag_prefix: str, scalars: dict, step: int = None):
        if self.writer is None:
            return
        s = step if step is not None else self.global_step
        for k, v in scalars.items():
            self.writer.add_scalar(f"{tag_prefix}/{k}", v, s)

    def log_scalar(self, tag: str, value: float, step: int = None):
        if self.writer is None:
            return
        s = step if step is not None else self.global_step
        self.writer.add_scalar(tag, value, s)

    def step(self):
        self.global_step += 1

    def flush(self):
        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()


class CSVLogger:
    """Append-mode CSV logger that auto-writes headers on first row per stage."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._files = {}
        self._writers = {}

    def log(self, stage: str, row: dict):
        if stage not in self._files:
            path = os.path.join(self.log_dir, f"{stage}_losses.csv")
            f = open(path, "a", newline="")
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if f.tell() == 0:
                writer.writeheader()
            self._files[stage] = f
            self._writers[stage] = writer
        self._writers[stage].writerow(row)
        self._files[stage].flush()

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()
        self._writers.clear()


# ---------------------------------------------------------------------------
# Stage 1: Pre-training with Multi-Token Prediction
# ---------------------------------------------------------------------------

def train_mtp(model, train_loader, val_loader, cfg, tb: TBLogger,
              is_fsdp=False, csv_log: CSVLogger = None):
    print0("\n" + "=" * 60)
    print0("Stage 1: Pre-training with Multi-Token Prediction (MTP)")
    print0("=" * 60)

    device = torch.device("musa", local_rank())
    dtype = get_dtype(cfg.train.dtype)

    if not is_fsdp:
        model = model.to(device=device, dtype=dtype)

    _trace("train_mtp: before AdamW")
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.mtp_lr,
        weight_decay=cfg.train.mtp_weight_decay,
    )
    _trace("train_mtp: after AdamW")
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.train.mtp_epochs, eta_min=cfg.train.mtp_lr * 0.01
    )

    best_val_loss = float("inf")
    os.makedirs(cfg.train.save_dir, exist_ok=True)

    _trace(f"train_mtp: entering epoch loop  n_batches={len(train_loader)}")
    for epoch in range(cfg.train.mtp_epochs):
        model.train()
        if hasattr(train_loader, "sampler") and isinstance(
            train_loader.sampler, DistributedSampler
        ):
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_metrics = {"ce_loss": 0.0, "refine_loss": 0.0}
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"MTP Epoch {epoch+1}/{cfg.train.mtp_epochs}",
            disable=not is_main(),
        )
        for batch_idx, batch in enumerate(pbar):
            _trace(f"train_mtp: epoch {epoch+1} batch {batch_idx}  to_device")
            batch = to_device(batch, device, dtype)

            # Log first batch shapes once
            if epoch == 0 and batch_idx == 0:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        _trace(f"  batch['{k}']: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")

            _trace(f"train_mtp: epoch {epoch+1} batch {batch_idx}  forward")
            result = model(batch, mode="mtp")

            _trace(f"train_mtp: epoch {epoch+1} batch {batch_idx}  mtp_loss")
            loss, metrics = mtp_loss(
                result, batch["target_ids"], n_heads=cfg.model.n_mtp_heads
            )

            _trace(f"train_mtp: epoch {epoch+1} batch {batch_idx}  zero_grad")
            optimizer.zero_grad()

            _trace(f"train_mtp: epoch {epoch+1} batch {batch_idx}  backward")
            loss.backward()

            _trace(f"train_mtp: epoch {epoch+1} batch {batch_idx}  clip_grad_norm")
            grad_norm = clip_grad_norm(model, 1.0, is_fsdp)

            _trace(f"train_mtp: epoch {epoch+1} batch {batch_idx}  optimizer.step")
            optimizer.step()

            _trace(f"train_mtp: epoch {epoch+1} batch {batch_idx}  done  loss={loss.item():.4f}")

            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] += v
            n_batches += 1

            tb.log_scalars("mtp_train_step", {
                "total_loss": loss.item(),
                "ce_loss": metrics["ce_loss"],
                "refine_loss": metrics["refine_loss"],
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "lr": optimizer.param_groups[0]["lr"],
            })
            tb.step()

            if is_main():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_metrics = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}

        val_loss, val_metrics = evaluate_model(
            model, val_loader, cfg, mode="mtp", is_fsdp=is_fsdp
        )

        tb.log_scalars("mtp_train_epoch", {
            "avg_loss": avg_loss,
            **{f"avg_{k}": v for k, v in avg_metrics.items()},
        }, step=epoch)
        tb.log_scalars("mtp_val_epoch", {
            "loss": val_loss,
            "hitrate_l1": val_metrics["hitrate_l1"],
            "hitrate_full": val_metrics["hitrate_full"],
        }, step=epoch)
        tb.log_scalar("lr/mtp", optimizer.param_groups[0]["lr"], step=epoch)
        tb.flush()

        if csv_log is not None and is_main():
            csv_log.log("mtp", {
                "epoch": epoch + 1,
                "train_loss": f"{avg_loss:.6f}",
                "train_ce": f"{avg_metrics['ce_loss']:.6f}",
                "train_refine": f"{avg_metrics['refine_loss']:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_hitrate_l1": f"{val_metrics['hitrate_l1']:.6f}",
                "val_hitrate_full": f"{val_metrics['hitrate_full']:.6f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.8f}",
            })

        print0(
            f"  Train Loss: {avg_loss:.4f} | "
            f"CE: {avg_metrics['ce_loss']:.4f} | "
            f"Refine: {avg_metrics['refine_loss']:.4f}"
        )
        print0(
            f"  Val Loss:   {val_loss:.4f} | "
            f"HitRate@L1: {val_metrics['hitrate_l1']:.4f} | "
            f"HitRate@Full: {val_metrics['hitrate_full']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                os.path.join(cfg.train.save_dir, "mtp_best.pt"),
                is_fsdp=is_fsdp,
            )
            print0("  -> Saved best MTP model")

        torch.musa.empty_cache()
        _trace(f"train_mtp: epoch {epoch+1} done, cache cleared")

    print0(f"MTP training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Stage 2: Value-Aware Fine-Tuning (VAFT)
# ---------------------------------------------------------------------------

def train_vaft(model, train_loader, val_loader, cfg, tb: TBLogger,
               is_fsdp=False, csv_log: CSVLogger = None):
    print0("\n" + "=" * 60)
    print0("Stage 2: Value-Aware Fine-Tuning (VAFT)")
    print0("=" * 60)

    device = torch.device("musa", local_rank())
    dtype = get_dtype(cfg.train.dtype)

    if not is_fsdp:
        model = model.to(device=device, dtype=dtype)

    optimizer = AdamW(model.parameters(), lr=cfg.train.vaft_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.train.vaft_epochs, eta_min=cfg.train.vaft_lr * 0.01
    )

    best_val_loss = float("inf")

    for epoch in range(cfg.train.vaft_epochs):
        model.train()
        if hasattr(train_loader, "sampler") and isinstance(
            train_loader.sampler, DistributedSampler
        ):
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_metrics = {"ce_loss": 0.0, "value_loss": 0.0}
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"VAFT Epoch {epoch+1}/{cfg.train.vaft_epochs}",
            disable=not is_main(),
        )
        for batch in pbar:
            batch = to_device(batch, device, dtype)

            result = model(batch, mode="vaft")
            loss, metrics = vaft_loss(
                result, batch["target_ids"],
                batch["target_value"], batch["target_action"],
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm(model, 1.0, is_fsdp)
            optimizer.step()

            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] += v
            n_batches += 1

            tb.log_scalars("vaft_train_step", {
                "total_loss": loss.item(),
                "ce_loss": metrics["ce_loss"],
                "value_loss": metrics["value_loss"],
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "lr": optimizer.param_groups[0]["lr"],
            })
            tb.step()

            if is_main():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_metrics = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}

        val_loss, val_metrics = evaluate_model(
            model, val_loader, cfg, mode="vaft", is_fsdp=is_fsdp
        )

        tb.log_scalars("vaft_train_epoch", {
            "avg_loss": avg_loss,
            **{f"avg_{k}": v for k, v in avg_metrics.items()},
        }, step=epoch)
        tb.log_scalars("vaft_val_epoch", {
            "loss": val_loss,
            "hitrate_l1": val_metrics["hitrate_l1"],
            "hitrate_full": val_metrics["hitrate_full"],
        }, step=epoch)
        tb.log_scalar("lr/vaft", optimizer.param_groups[0]["lr"], step=epoch)
        tb.flush()

        if csv_log is not None and is_main():
            csv_log.log("vaft", {
                "epoch": epoch + 1,
                "train_loss": f"{avg_loss:.6f}",
                "train_ce": f"{avg_metrics['ce_loss']:.6f}",
                "train_value": f"{avg_metrics['value_loss']:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_hitrate_l1": f"{val_metrics['hitrate_l1']:.6f}",
                "val_hitrate_full": f"{val_metrics['hitrate_full']:.6f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.8f}",
            })

        print0(
            f"  Train Loss: {avg_loss:.4f} | "
            f"CE: {avg_metrics['ce_loss']:.4f} | "
            f"Value: {avg_metrics['value_loss']:.4f}"
        )
        print0(
            f"  Val Loss:   {val_loss:.4f} | "
            f"HitRate@L1: {val_metrics['hitrate_l1']:.4f} | "
            f"HitRate@Full: {val_metrics['hitrate_full']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                os.path.join(cfg.train.save_dir, "vaft_best.pt"),
                is_fsdp=is_fsdp,
            )
            print0("  -> Saved best VAFT model")

        torch.musa.empty_cache()

    print0(f"VAFT training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Stage 3: Post-training with HEPO
# ---------------------------------------------------------------------------

def _compute_batch_popularity(semantic_ids, token_types, action_types, n_levels):
    B = semantic_ids.shape[0]
    popularity = []

    for b in range(B):
        i_mask = (token_types[b] == 3)
        i_positions = i_mask.nonzero(as_tuple=False).squeeze(-1)
        n_i = len(i_positions)

        actions_b = action_types[b][:n_i]

        pop_per_level = {}
        for lvl in range(n_levels):
            codes = semantic_ids[b, i_positions, lvl]
            positive = (actions_b > 0)
            pos_codes = codes[positive]

            if len(pos_codes) == 0:
                pop_per_level[lvl] = {}
                continue

            unique_codes, counts = pos_codes.unique(return_counts=True)
            total = counts.sum().item()
            pop_per_level[lvl] = {
                c.item(): cnt.item() / total
                for c, cnt in zip(unique_codes, counts)
            }

        popularity.append(pop_per_level)

    return popularity


def compute_process_rewards(codes, popularity, all_codes_per_level,
                            terminal_rewards, alpha=0.1):
    B, K, n_levels = codes.shape
    device = codes.device
    rewards = torch.zeros(B, K, n_levels, device=device, dtype=terminal_rewards.dtype)

    for b in range(B):
        for lvl in range(n_levels):
            if lvl < n_levels - 1:
                pop_b = popularity[b].get(lvl, {})
                legal_codes = all_codes_per_level.get(lvl, set())
                avg_pop = (
                    sum(pop_b.get(c, 0.0) for c in legal_codes)
                    / max(len(legal_codes), 1)
                )

                for k_idx in range(K):
                    code_val = codes[b, k_idx, lvl].item()
                    p_code = pop_b.get(code_val, 0.0)
                    delta = p_code - avg_pop
                    rewards[b, k_idx, lvl] = alpha * max(0.0, delta)
            else:
                rewards[b, :, lvl] = terminal_rewards[b, :]

    return rewards


def train_hepo(model, train_loader, val_loader, cfg, tb: TBLogger,
               extra_data=None, is_fsdp=False, csv_log: CSVLogger = None):
    print0("\n" + "=" * 60)
    print0("Stage 3: Post-training with HEPO")
    print0("=" * 60)

    device = torch.device("musa", local_rank())
    dtype = get_dtype(cfg.train.dtype)

    if not is_fsdp:
        model = model.to(device=device, dtype=dtype)

    n_levels = cfg.model.n_semantic_levels
    gamma = cfg.train.gamma
    lam = cfg.train.lam

    level_coeffs = [(lvl + 1) / n_levels for lvl in range(n_levels)]

    all_codes_per_level = extra_data.get("all_codes_per_level", {}) if extra_data else {}
    item2sid = extra_data.get("item2sid", {}) if extra_data else {}
    all_items = extra_data.get("all_items", []) if extra_data else []

    policy_params = []
    value_params = []
    for name, param in model.named_parameters():
        if "hte" in name:
            value_params.append(param)
        else:
            policy_params.append(param)

    policy_optimizer = AdamW(policy_params, lr=cfg.train.hepo_lr_policy)
    value_optimizer = AdamW(value_params, lr=cfg.train.hepo_lr_value)

    best_val_loss = float("inf")

    for epoch in range(cfg.train.hepo_epochs):
        model.train()
        if hasattr(train_loader, "sampler") and isinstance(
            train_loader.sampler, DistributedSampler
        ):
            train_loader.sampler.set_epoch(epoch)

        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_avg_reward = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"HEPO Epoch {epoch+1}/{cfg.train.hepo_epochs}",
            disable=not is_main(),
        )
        for batch in pbar:
            batch = to_device(batch, device, dtype)

            if cfg.train.arr_enabled and item2sid:
                arr_batch = generate_arr_samples(
                    batch, item2sid, all_items,
                    ratio=cfg.train.arr_synthetic_ratio,
                    n_levels=n_levels,
                )
                batch = merge_batches(batch, arr_batch)

            model.eval()
            gen_result_full = model(
                batch, mode="hepo_candidates",
                n_candidates=cfg.train.n_candidates,
            )
            model.train()

            cand_codes = gen_result_full["codes"]
            old_logprobs = gen_result_full["logprobs"]
            pred_values = gen_result_full["values"]

            B, K, _ = cand_codes.shape
            target_ids = batch["target_ids"]
            target_value = batch["target_value"]

            popularity = _compute_batch_popularity(
                batch["semantic_ids"], batch["token_types"],
                batch["action_types"], n_levels,
            )

            terminal_reward = (
                -torch.abs(pred_values - target_value.unsqueeze(1))
                + (cand_codes[:, :, -1] == target_ids[:, -1].unsqueeze(1)).to(dtype)
            )

            rewards = compute_process_rewards(
                cand_codes, popularity, all_codes_per_level,
                terminal_reward, alpha=cfg.train.hepo_alpha,
            )

            returns = torch.zeros_like(rewards)
            for lvl in reversed(range(n_levels)):
                if lvl == n_levels - 1:
                    returns[:, :, lvl] = rewards[:, :, lvl]
                else:
                    returns[:, :, lvl] = rewards[:, :, lvl] + gamma * returns[:, :, lvl + 1]

            hepo_batch = {**batch, "cand_codes": cand_codes}
            hepo_result = model(hepo_batch, mode="hepo_train")

            new_logprobs = hepo_result["new_logprobs"]
            value_preds = hepo_result["value_preds"]

            values_det = value_preds.detach()
            advantages = torch.zeros_like(rewards)
            for k_idx in range(K):
                deltas = []
                for lvl in range(n_levels - 1):
                    delta = (
                        rewards[:, k_idx, lvl]
                        + gamma * values_det[:, k_idx, lvl + 1]
                        - values_det[:, k_idx, lvl]
                    )
                    deltas.append(delta)

                gae = torch.zeros(B, device=device, dtype=dtype)
                for lvl in reversed(range(n_levels - 1)):
                    gae = deltas[lvl] + gamma * lam * gae
                    advantages[:, k_idx, lvl] = gae

                advantages[:, k_idx, -1] = rewards[:, k_idx, -1]

            final_adv = advantages[:, :, -1]
            mu = final_adv.mean(dim=1, keepdim=True)
            sigma = final_adv.std(dim=1, keepdim=True) + 1e-8
            advantages[:, :, -1] = (final_adv - mu) / sigma

            ratio = torch.exp(new_logprobs - old_logprobs.detach())
            eps = cfg.train.clip_eps
            clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
            adv = advantages.detach()

            policy_loss = 0.0
            for lvl in range(n_levels):
                c_l = level_coeffs[lvl]
                lvl_loss = -torch.min(
                    ratio[:, :, lvl] * adv[:, :, lvl],
                    clipped[:, :, lvl] * adv[:, :, lvl],
                ).mean()
                policy_loss = policy_loss + c_l * lvl_loss

            value_loss = F.mse_loss(value_preds, returns.detach())

            total_loss = policy_loss + 0.5 * value_loss

            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            total_loss.backward()
            grad_norm = clip_grad_norm(model, 1.0, is_fsdp)
            policy_optimizer.step()
            value_optimizer.step()

            avg_reward = rewards.sum(dim=-1).mean().item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_avg_reward += avg_reward
            n_batches += 1

            tb.log_scalars("hepo_train_step", {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "total_loss": total_loss.item(),
                "avg_reward": avg_reward,
                "avg_ratio": ratio.mean().item(),
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            })
            tb.step()

            if is_main():
                pbar.set_postfix(
                    policy=f"{policy_loss.item():.4f}",
                    value=f"{value_loss.item():.4f}",
                    reward=f"{avg_reward:.4f}",
                )

        avg_pl = epoch_policy_loss / max(n_batches, 1)
        avg_vl = epoch_value_loss / max(n_batches, 1)
        avg_rw = epoch_avg_reward / max(n_batches, 1)

        val_loss, val_metrics = evaluate_model(
            model, val_loader, cfg, mode="mtp", is_fsdp=is_fsdp
        )

        tb.log_scalars("hepo_train_epoch", {
            "avg_policy_loss": avg_pl,
            "avg_value_loss": avg_vl,
            "avg_reward": avg_rw,
        }, step=epoch)
        tb.log_scalars("hepo_val_epoch", {
            "loss": val_loss,
            "hitrate_l1": val_metrics["hitrate_l1"],
            "hitrate_full": val_metrics["hitrate_full"],
        }, step=epoch)
        tb.flush()

        if csv_log is not None and is_main():
            csv_log.log("hepo", {
                "epoch": epoch + 1,
                "policy_loss": f"{avg_pl:.6f}",
                "value_loss": f"{avg_vl:.6f}",
                "avg_reward": f"{avg_rw:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_hitrate_l1": f"{val_metrics['hitrate_l1']:.6f}",
                "val_hitrate_full": f"{val_metrics['hitrate_full']:.6f}",
            })

        print0(
            f"  Policy Loss: {avg_pl:.4f} | Value Loss: {avg_vl:.4f} | "
            f"Avg Reward: {avg_rw:.4f} | "
            f"Val HitRate@L1: {val_metrics['hitrate_l1']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                os.path.join(cfg.train.save_dir, "hepo_best.pt"),
                is_fsdp=is_fsdp,
            )
            print0("  -> Saved best HEPO model")

        torch.musa.empty_cache()

    print0(f"HEPO training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, val_loader, cfg, mode="mtp", is_fsdp=False):
    device = torch.device("musa", local_rank())
    dtype = get_dtype(cfg.train.dtype)
    model.eval()

    total_loss = 0.0
    total_l1_hits = 0
    total_full_hits = 0
    total_samples = 0

    for batch in val_loader:
        batch = to_device(batch, device, dtype)

        result = model(batch, mode="mtp")

        if mode == "vaft":
            loss, _ = vaft_loss(
                result, batch["target_ids"],
                batch["target_value"], batch["target_action"],
            )
        else:
            loss, _ = mtp_loss(
                result, batch["target_ids"], n_heads=cfg.model.n_mtp_heads,
            )

        B = batch["target_ids"].shape[0]
        total_loss += loss.item() * B

        pred_codes = result["all_logits"][0].argmax(dim=-1)
        target = batch["target_ids"]

        total_l1_hits += (pred_codes[:, 0] == target[:, 0]).sum().item()
        total_full_hits += (pred_codes == target).all(dim=1).sum().item()
        total_samples += B

    if is_dist():
        stats = torch.tensor(
            [total_loss, total_l1_hits, total_full_hits, total_samples],
            device=device, dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_l1_hits, total_full_hits, total_samples = stats.tolist()

    avg_loss = total_loss / max(total_samples, 1)
    l1_hitrate = total_l1_hits / max(total_samples, 1)
    full_hitrate = total_full_hits / max(total_samples, 1)

    model.train()
    return avg_loss, {
        "hitrate_l1": l1_hitrate,
        "hitrate_full": full_hitrate,
    }


# ---------------------------------------------------------------------------
# Data preparation (runs on all ranks, but downloads on rank 0 only)
# ---------------------------------------------------------------------------

def prepare_data(cfg, is_distributed=False):
    print0("Preparing data...")

    if cfg.data.dataset == "synthetic":
        print0("Using synthetic dataset")
        df, item_meta, item_embeddings, _ = generate_synthetic_data(cfg.data)
    else:
        print0(f"Loading Amazon {cfg.data.amazon_category} reviews...")
        try:
            df, item_meta = load_amazon_reviews(cfg.data)
            print0(f"  Loaded {len(df)} interactions, {df['item'].nunique()} items")
            n_items = df["item"].nunique()
            item_embeddings = np.random.randn(n_items, cfg.data.item_embed_dim).astype(
                np.float32
            )
            item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        except Exception as e:
            print0(f"  Failed to load Amazon data: {e}")
            print0("  Falling back to synthetic dataset")
            cfg.data.dataset = "synthetic"
            df, item_meta, item_embeddings, _ = generate_synthetic_data(cfg.data)

    unique_items = sorted(df["item"].unique())
    item2idx = {item: idx for idx, item in enumerate(unique_items)}

    cfg.model.n_items = len(unique_items)
    cfg.model.n_users = df["user"].nunique()
    print0(f"  {cfg.model.n_users} users, {cfg.model.n_items} items")

    if len(item_embeddings) < len(unique_items):
        extra = np.random.randn(
            len(unique_items) - len(item_embeddings), cfg.data.item_embed_dim
        ).astype(np.float32)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True) + 1e-8
        item_embeddings = np.vstack([item_embeddings, extra])
    item_embeddings = item_embeddings[:len(unique_items)]

    tokenizer_path = os.path.join(cfg.train.save_dir, "tokenizer.pt")
    os.makedirs(cfg.train.save_dir, exist_ok=True)

    # Keep tokenizer on CPU for data preparation.  encode_all() is only
    # 579 items — sub-second on CPU — and avoids 8 ranks concurrently
    # hitting the MUSA driver during data prep (intermittent segfaults).
    _trace("prepare_data: before tokenizer creation (CPU)")
    tokenizer = RQKMeansPlus(cfg.tokenizer)
    _trace("prepare_data: tokenizer created on CPU")

    if os.path.exists(tokenizer_path):
        print0("Loading existing tokenizer...")
        tokenizer.load(tokenizer_path)
        _trace("prepare_data: tokenizer loaded")
    else:
        print0("Training RQ-KMeans+ tokenizer (on MUSA rank 0)...")
        if is_main():
            tok_device = torch.device("musa", local_rank())
            tokenizer = tokenizer.to(tok_device)
            tokenizer.fit(item_embeddings, cfg.tokenizer)
            tokenizer.save(tokenizer_path)
            tokenizer = tokenizer.cpu()
        _trace("prepare_data: tokenizer trained/saved")
        if is_distributed:
            dist.barrier()
            if not is_main():
                tokenizer.load(tokenizer_path)
        _trace("prepare_data: all ranks have tokenizer")

    if is_distributed:
        _trace("prepare_data: barrier before encode_all")
        dist.barrier()
        _trace("prepare_data: barrier done")

    _trace("prepare_data: encode_all (CPU)")
    all_semantic_ids = tokenizer.encode_all(item_embeddings)
    _trace("prepare_data: encode_all done")

    item2sid = {
        item: all_semantic_ids[item2idx[item]].tolist()
        for item in unique_items
        if item in item2idx
    }
    _trace("prepare_data: item2sid built")

    user_seqs = build_sequences(df, item_meta, cfg.data)
    _trace("prepare_data: build_sequences done")
    print0(f"  Built {len(user_seqs)} user sequences")

    _trace("prepare_data: before create_dataloaders_distributed")
    train_loader, val_loader = create_dataloaders_distributed(
        user_seqs, item2sid, cfg, is_distributed
    )
    print0(
        f"  Train: {len(train_loader.dataset)} samples, "
        f"Val: {len(val_loader.dataset)} samples"
    )
    _trace("prepare_data: dataloaders created")

    trie = SemanticTrie.build_from_items(item2sid)
    all_codes_per_level = get_all_codes_per_level(
        item2sid, cfg.model.n_semantic_levels
    )
    user_code_popularity = build_user_code_popularity(
        user_seqs, item2sid, cfg.model.n_semantic_levels
    )

    extra_data = {
        "trie": trie,
        "all_codes_per_level": all_codes_per_level,
        "user_code_popularity": user_code_popularity,
        "item2sid": item2sid,
        "all_items": list(item2sid.keys()),
    }

    return train_loader, val_loader, extra_data


def create_dataloaders_distributed(user_seqs, item2sid, cfg, is_distributed):
    """Create DataLoaders with DistributedSampler when running multi-GPU."""
    from data_utils import GPRDataset

    rng = random.Random(cfg.train.seed)
    user_seqs = list(user_seqs)
    rng.shuffle(user_seqs)

    split = int(len(user_seqs) * 0.9)
    train_seqs = user_seqs[:split]
    val_seqs = user_seqs[split:]

    train_ds = GPRDataset(
        train_seqs, item2sid,
        n_levels=cfg.model.n_semantic_levels,
        max_seq_len=cfg.data.max_seq_len,
        n_organic=cfg.data.n_organic_per_sample,
        d_user=cfg.model.d_user,
        d_env=cfg.model.d_env,
        is_train=True,
    )
    val_ds = GPRDataset(
        val_seqs, item2sid,
        n_levels=cfg.model.n_semantic_levels,
        max_seq_len=cfg.data.max_seq_len,
        n_organic=cfg.data.n_organic_per_sample,
        d_user=cfg.model.d_user,
        d_env=cfg.model.d_env,
        is_train=False,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPR Training Pipeline (FSDP/MUSA)")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "amazon"])
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "mtp", "vaft", "hepo"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--mtp_epochs", type=int, default=None)
    parser.add_argument("--vaft_epochs", type=int, default=None)
    parser.add_argument("--hepo_epochs", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--no_fsdp", action="store_true",
                        help="Disable FSDP (single-GPU mode)")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    is_distributed = setup_distributed()
    use_fsdp = is_distributed and not args.no_fsdp

    cfg = GPRConfig()
    cfg.data.dataset = args.dataset
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.mtp_epochs:
        cfg.train.mtp_epochs = args.mtp_epochs
    if args.vaft_epochs:
        cfg.train.vaft_epochs = args.vaft_epochs
    if args.hepo_epochs:
        cfg.train.hepo_epochs = args.hepo_epochs
    cfg = cfg.sync()

    set_seed(cfg.train.seed + rank(), deterministic=cfg.train.deterministic)

    run_name = args.run_name or (
        f"gpr_musa_fsdp{world_size()}_{args.dataset}_{args.stage}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    tb_log_dir = os.path.join(args.log_dir, run_name)
    if is_main():
        os.makedirs(tb_log_dir, exist_ok=True)
    tb = TBLogger(tb_log_dir)
    csv_log = CSVLogger(tb_log_dir) if is_main() else None

    if is_main():
        config_path = os.path.join(tb_log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({
                "dataset": cfg.data.dataset,
                "stage": args.stage,
                "world_size": world_size(),
                "use_fsdp": use_fsdp,
                "batch_size_per_gpu": cfg.train.batch_size,
                "effective_batch_size": cfg.train.batch_size * world_size(),
                "d_model": cfg.model.d_model,
                "n_heads": cfg.model.n_heads,
                "d_ff": cfg.model.d_ff,
                "n_layers_hsd": cfg.model.n_layers_hsd,
                "n_layers_ptd": cfg.model.n_layers_ptd,
                "n_semantic_levels": cfg.model.n_semantic_levels,
                "codebook_size": cfg.model.codebook_size,
                "max_seq_len": cfg.model.max_seq_len,
                "n_mtp_heads": cfg.model.n_mtp_heads,
                "mtp_epochs": cfg.train.mtp_epochs,
                "mtp_lr": cfg.train.mtp_lr,
                "vaft_epochs": cfg.train.vaft_epochs,
                "vaft_lr": cfg.train.vaft_lr,
                "hepo_epochs": cfg.train.hepo_epochs,
                "activation_checkpointing": cfg.train.activation_checkpointing,
            }, f, indent=2)

    print0(f"World size: {world_size()}")
    print0(f"FSDP: {use_fsdp}")
    print0(f"Device: musa:{local_rank()}")
    print0(f"Dtype: {cfg.train.dtype}")
    print0(f"Model: d={cfg.model.d_model}, HSD layers={cfg.model.n_layers_hsd}, "
           f"PTD layers={cfg.model.n_layers_ptd}")
    print0(f"Batch size per GPU: {cfg.train.batch_size}, "
           f"Effective: {cfg.train.batch_size * world_size()}")
    print0(f"Dataset: {cfg.data.dataset}")
    print0(f"TensorBoard: {tb_log_dir}")

    _trace("main: before prepare_data")
    train_loader, val_loader, extra_data = prepare_data(cfg, is_distributed)
    _trace("main: after prepare_data")

    _trace("main: before GPR()")
    model = GPR(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print0(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    _trace("main: GPR created on CPU")

    tb.log_scalar("model/n_params_M", n_params / 1e6, step=0)
    tb.log_scalar("model/n_layers_hsd", cfg.model.n_layers_hsd, step=0)
    tb.log_scalar("model/d_model", cfg.model.d_model, step=0)

    if use_fsdp:
        _trace("main: before wrap_model_fsdp")
        model = wrap_model_fsdp(model, cfg)
        _trace("main: after wrap_model_fsdp")
        print0("Model wrapped with FSDP")
        if cfg.train.activation_checkpointing:
            print0("Activation checkpointing enabled for HSDBlock")
    else:
        device = torch.device("musa", local_rank())
        dtype = get_dtype(cfg.train.dtype)
        _trace("main: before model.to(device)")
        model = model.to(device=device, dtype=dtype)
        _trace("main: after model.to(device)")

    if args.resume:
        print0(f"Loading checkpoint: {args.resume}")
        _trace("main: before load_checkpoint")
        load_checkpoint(model, args.resume, is_fsdp=use_fsdp)
        _trace("main: after load_checkpoint")

    t0 = time.time()

    if args.stage in ("all", "mtp"):
        _trace("main: before train_mtp")
        model = train_mtp(model, train_loader, val_loader, cfg, tb,
                          is_fsdp=use_fsdp, csv_log=csv_log)
        _trace("main: after train_mtp")

    if args.stage in ("all", "vaft"):
        if args.stage == "vaft" and args.resume is None:
            mtp_path = os.path.join(cfg.train.save_dir, "mtp_best.pt")
            if os.path.exists(mtp_path):
                print0(f"Loading MTP checkpoint: {mtp_path}")
                load_checkpoint(model, mtp_path, is_fsdp=use_fsdp)
        model = train_vaft(model, train_loader, val_loader, cfg, tb,
                           is_fsdp=use_fsdp, csv_log=csv_log)

    if args.stage in ("all", "hepo"):
        if args.stage == "hepo" and args.resume is None:
            vaft_path = os.path.join(cfg.train.save_dir, "vaft_best.pt")
            if os.path.exists(vaft_path):
                print0(f"Loading VAFT checkpoint: {vaft_path}")
                load_checkpoint(model, vaft_path, is_fsdp=use_fsdp)
        model = train_hepo(model, train_loader, val_loader, cfg, tb,
                           extra_data=extra_data, is_fsdp=use_fsdp,
                           csv_log=csv_log)

    elapsed = time.time() - t0
    tb.log_scalar("timing/total_minutes", elapsed / 60, step=0)
    tb.close()
    if csv_log:
        csv_log.close()

    final_path = os.path.join(cfg.train.save_dir, "gpr_final.pt")
    save_checkpoint(model, final_path, is_fsdp=use_fsdp)
    print0(f"\nTotal training time: {elapsed / 60:.1f} minutes")
    print0(f"Final model saved to {final_path}")
    print0(f"TensorBoard logs at: {tb_log_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        rank = int(os.environ.get("RANK", -1))
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"[rank{rank}] FATAL ERROR:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr, flush=True)
        raise
