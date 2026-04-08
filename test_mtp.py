#!/usr/bin/env python3
"""
MTP (stage 1) tests aligned with train_mtp() in train.py.

What is covered
  - Forward: ``model(batch, mode="mtp")`` (same as training).
  - Loss: ``mtp_loss(result, batch["target_ids"], n_heads=...)`` (same call as training).
  - Autograd: ``loss.backward()`` over the full graph (CE + refine + paths that
    receive gradients in MTP).

What is *not* the full MTP *training run*
  - No multi-epoch loop, DataLoader, validation, TensorBoard, or checkpointing.
  - Uses a small ``ModelConfig`` for speed (not the full default GPR width/depth).

The test ``test_mtp_one_step_matches_train_mtp_loop`` runs a single iteration using
the same sequence as the training loop: ``to_device`` → forward → ``mtp_loss`` →
``zero_grad`` → ``backward`` → ``clip_grad_norm_(..., 1.0)`` → ``optimizer.step``.

Run:
  python test_mtp.py
  python -m pytest test_mtp.py -v
"""

from __future__ import annotations

import sys
import types
import unittest

import torch
import torch.nn.functional as F

# config.py imports torch_musa; allow running tests on CPU-only envs without the wheel.
try:
    import torch_musa  # noqa: F401
except ModuleNotFoundError:
    sys.modules.setdefault("torch_musa", types.ModuleType("torch_musa"))
    if not hasattr(torch, "musa"):
        def _noop(*_a, **_k):
            return None

        torch.musa = types.SimpleNamespace(
            is_available=lambda: False,
            manual_seed=_noop,
            manual_seed_all=_noop,
            synchronize=_noop,
        )

from config import GPRConfig, ModelConfig  # noqa: E402
from model import GPR, mtp_loss  # noqa: E402


def _mtp_device() -> torch.device:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.device("musa:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _tiny_model_config() -> ModelConfig:
    """Small config for fast tests; fields must stay consistent across HSD/PTD/HTE."""
    return ModelConfig(
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_layers_hsd=2,
        n_layers_ptd=1,
        n_thinking_tokens=2,
        n_refining_steps=3,
        n_semantic_levels=3,
        codebook_size=32,
        max_seq_len=16,
        n_token_types=4,
        dropout=0.0,
        n_items=128,
        n_users=64,
        d_user=8,
        d_env=4,
        n_mtp_heads=2,
        n_mor_recursions=1,
        n_llm_thought_tokens=2,
    )


def _make_mtp_batch(
    *,
    batch_size: int,
    max_seq_len: int,
    n_levels: int,
    codebook_size: int,
    d_user: int,
    d_env: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """
    Minimal batch matching GPRDataset / DataLoader stacking and train.to_device().
    """
    L = max_seq_len
    semantic_ids = torch.zeros(batch_size, L, n_levels, dtype=torch.long, device=device)
    token_types = torch.zeros(batch_size, L, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Effective length excludes right-side padding (same convention as dataset).
        seq_len = 10 + b % 5  # 10..14
        # [U, O, O, E, I, I, ...] then pads with type 0
        types = [0, 1, 1, 2, 3, 3, 3, 3]
        types = types + [0] * (L - len(types))
        types = types[:L]
        token_types[b] = torch.tensor(types, device=device)

        # Embedding is (codebook_size + 1); non-padding codes in [1, codebook_size]
        sem = torch.randint(1, codebook_size + 1, (L, n_levels), device=device)
        sem[token_types[b] == 0] = 0
        semantic_ids[b] = sem

    user_features = torch.randn(batch_size, d_user, device=device, dtype=dtype)
    env_features = torch.randn(batch_size, d_env, device=device, dtype=dtype)
    seq_len = torch.tensor([10 + b % 5 for b in range(batch_size)], device=device)
    # PTD heads output ``codebook_size`` logits → targets must be in [0, codebook_size)
    target_ids = torch.randint(0, codebook_size, (batch_size, n_levels), device=device)

    return {
        "semantic_ids": semantic_ids,
        "token_types": token_types,
        "user_features": user_features,
        "env_features": env_features,
        "seq_len": seq_len,
        "target_ids": target_ids,
    }


class TestMTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = _mtp_device()
        cls.cfg = _tiny_model_config()

    def test_mtp_forward_structure(self):
        model = GPR(self.cfg).to(self.device, dtype=torch.float32)
        model.eval()
        batch = _make_mtp_batch(
            batch_size=2,
            max_seq_len=self.cfg.max_seq_len,
            n_levels=self.cfg.n_semantic_levels,
            codebook_size=self.cfg.codebook_size,
            d_user=self.cfg.d_user,
            d_env=self.cfg.d_env,
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            out = model(batch, mode="mtp")

        self.assertIn("all_logits", out)
        self.assertEqual(len(out["all_logits"]), self.cfg.n_mtp_heads)
        B = batch["semantic_ids"].shape[0]
        for logits in out["all_logits"]:
            self.assertEqual(
                logits.shape,
                (B, self.cfg.n_semantic_levels, self.cfg.codebook_size),
            )
        self.assertEqual(out["level_values"].shape, (B, self.cfg.n_semantic_levels))
        self.assertEqual(out["final_value"].shape, (B, 1))
        self.assertEqual(out["pred_codes"].shape, (B, self.cfg.n_semantic_levels))

    def test_mtp_loss_finite(self):
        model = GPR(self.cfg).to(self.device, dtype=torch.float32)
        model.train()
        batch = _make_mtp_batch(
            batch_size=2,
            max_seq_len=self.cfg.max_seq_len,
            n_levels=self.cfg.n_semantic_levels,
            codebook_size=self.cfg.codebook_size,
            d_user=self.cfg.d_user,
            d_env=self.cfg.d_env,
            device=self.device,
            dtype=torch.float32,
        )
        out = model(batch, mode="mtp")
        loss, metrics = mtp_loss(out, batch["target_ids"], n_heads=self.cfg.n_mtp_heads)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIn("ce_loss", metrics)
        self.assertIn("refine_loss", metrics)

    def test_mtp_backward_no_nan_grads(self):
        model = GPR(self.cfg).to(self.device, dtype=torch.float32)
        model.train()
        batch = _make_mtp_batch(
            batch_size=2,
            max_seq_len=self.cfg.max_seq_len,
            n_levels=self.cfg.n_semantic_levels,
            codebook_size=self.cfg.codebook_size,
            d_user=self.cfg.d_user,
            d_env=self.cfg.d_env,
            device=self.device,
            dtype=torch.float32,
        )
        out = model(batch, mode="mtp")
        loss, _ = mtp_loss(out, batch["target_ids"], n_heads=self.cfg.n_mtp_heads)
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            n_nan = torch.isnan(p.grad).sum().item()
            self.assertEqual(n_nan, 0, msg=f"NaN grad in {name}")

    def test_mtp_cross_entropy_matches_manual(self):
        """Sanity-check mtp_loss head/level aggregation against explicit CE."""
        model = GPR(self.cfg).to(self.device, dtype=torch.float32)
        model.eval()
        batch = _make_mtp_batch(
            batch_size=3,
            max_seq_len=self.cfg.max_seq_len,
            n_levels=self.cfg.n_semantic_levels,
            codebook_size=self.cfg.codebook_size,
            d_user=self.cfg.d_user,
            d_env=self.cfg.d_env,
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            out = model(batch, mode="mtp")
        n_heads = self.cfg.n_mtp_heads
        n_levels = self.cfg.n_semantic_levels
        targets = batch["target_ids"]
        manual = 0.0
        w_h = 1.0 / n_heads
        for logits in out["all_logits"]:
            for lvl in range(n_levels):
                manual += w_h * F.cross_entropy(logits[:, lvl, :], targets[:, lvl])
        loss, metrics = mtp_loss(out, targets, n_heads=n_heads)
        expected = manual + 0.1 * out["refine_loss"]
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)
        self.assertAlmostEqual(metrics["ce_loss"], manual.item(), places=5)

    def test_mtp_bfloat16_if_supported(self):
        """Mirrors train_mtp (bf16 on accelerator) when the device supports it."""
        if self.device.type == "cpu":
            self.skipTest("bfloat16 MTP smoke test is for GPU/MUSA")
        if not torch.cuda.is_bf16_supported() and self.device.type != "musa":
            self.skipTest("bfloat16 not supported on this device")

        model = GPR(self.cfg).to(self.device, dtype=torch.bfloat16)
        model.train()
        batch = _make_mtp_batch(
            batch_size=2,
            max_seq_len=self.cfg.max_seq_len,
            n_levels=self.cfg.n_semantic_levels,
            codebook_size=self.cfg.codebook_size,
            d_user=self.cfg.d_user,
            d_env=self.cfg.d_env,
            device=self.device,
            dtype=torch.bfloat16,
        )
        out = model(batch, mode="mtp")
        loss, _ = mtp_loss(out, batch["target_ids"], n_heads=self.cfg.n_mtp_heads)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        if hasattr(torch, "musa") and self.device.type == "musa":
            torch.musa.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

    def test_mtp_one_step_matches_train_mtp_loop(self):
        """
        Mirror train_mtp()'s per-batch body (train.py): to_device, forward, mtp_loss,
        zero_grad, backward, grad clip 1.0, AdamW step. Verifies weights actually move.
        """
        from torch.optim import AdamW

        from train import get_dtype, to_device

        tcfg = GPRConfig().train
        dtype = get_dtype(tcfg.dtype)
        # Default train dtype is bf16 for accelerators; CPU often uses f32 in practice.
        if self.device.type == "cpu" and dtype == torch.bfloat16:
            dtype = torch.float32

        model = GPR(self.cfg).to(device=self.device, dtype=dtype)
        model.train()
        optimizer = AdamW(
            model.parameters(),
            lr=tcfg.mtp_lr,
            weight_decay=tcfg.mtp_weight_decay,
        )

        batch = _make_mtp_batch(
            batch_size=2,
            max_seq_len=self.cfg.max_seq_len,
            n_levels=self.cfg.n_semantic_levels,
            codebook_size=self.cfg.codebook_size,
            d_user=self.cfg.d_user,
            d_env=self.cfg.d_env,
            device=self.device,
            dtype=dtype,
        )
        device_str = str(self.device)
        batch = to_device(batch, device_str, dtype)

        before = {n: p.detach().float().clone() for n, p in model.named_parameters()}

        optimizer.zero_grad()
        result = model(batch, mode="mtp")
        loss, metrics = mtp_loss(
            result, batch["target_ids"], n_heads=self.cfg.n_mtp_heads
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        self.assertTrue(torch.isfinite(loss).item(), msg="loss must be finite")
        self.assertTrue(
            torch.isfinite(torch.as_tensor(grad_norm)).item(),
            msg="grad norm must be finite",
        )
        self.assertIn("ce_loss", metrics)
        self.assertIn("refine_loss", metrics)

        changed = False
        for n, p in model.named_parameters():
            delta = (p.float() - before[n]).abs().max().item()
            if delta > 0.0:
                changed = True
                break
        self.assertTrue(
            changed,
            msg="optimizer.step() should update at least one parameter (same as training)",
        )


if __name__ == "__main__":
    unittest.main()
