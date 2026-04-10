#!/usr/bin/env python3
"""
MUSA + FSDP diagnostic script.

Run step-by-step to isolate exactly where a native crash / SIGTERM occurs.

  Single-GPU:   python debug_fsdp.py
  Multi-GPU:    torchrun --nproc_per_node=8 debug_fsdp.py

Each step prints PASS / FAIL.  If the process is killed by a signal the
last printed step tells you where the crash happened.
"""

import os
import sys
import signal
import time

# ── global state for signal handler ──────────────────────────────────────
_current_step = "initialization"
_rank = int(os.environ.get("RANK", 0))


def _sig_handler(signum, frame):
    name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    print(f"\n[rank{_rank}] KILLED by {name} during: {_current_step}",
          file=sys.stderr, flush=True)
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _sig_handler)
try:
    signal.signal(signal.SIGSEGV, _sig_handler)
except Exception:
    pass


def log(msg):
    print(f"[rank{_rank}] {msg}", flush=True)


def step(name):
    global _current_step
    _current_step = name
    log(f"{'─'*50}")
    log(f"STEP: {name}")


def ok(detail=""):
    log(f"  ✓ PASS  {detail}")


def fail(detail=""):
    log(f"  ✗ FAIL  {detail}")


# ─────────────────────────────────────────────────────────────────────────
def main():
    global _rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    _rank = int(os.environ.get("RANK", 0))
    is_distributed = world_size > 1

    log(f"Python {sys.version}")
    log(f"rank={_rank}  local_rank={local_rank}  world_size={world_size}")

    # ── 1  import torch ──────────────────────────────────────────────────
    step("1. import torch")
    try:
        import torch
        ok(f"torch {torch.__version__}")
    except Exception as e:
        fail(str(e)); return

    # ── 2  import torch_musa ─────────────────────────────────────────────
    step("2. import torch_musa")
    try:
        import torch_musa
        ok("torch_musa imported")
    except Exception as e:
        fail(str(e)); return

    # ── 3  MUSA device count ─────────────────────────────────────────────
    step("3. MUSA device count")
    try:
        n_dev = torch.musa.device_count()
        if n_dev < world_size:
            fail(f"need {world_size} devices but only {n_dev} available"); return
        ok(f"{n_dev} device(s)")
    except Exception as e:
        fail(str(e)); return

    # ── 4  set_device ────────────────────────────────────────────────────
    step("4. set_device")
    try:
        torch.musa.set_device(local_rank)
        device = torch.device("musa", local_rank)
        ok(str(device))
    except Exception as e:
        fail(str(e)); return

    # ── 5  basic matmul on MUSA ──────────────────────────────────────────
    step("5. basic matmul fp32")
    try:
        a = torch.randn(8, 8, device=device)
        b = (a @ a.T).cpu()
        ok(f"shape={tuple(b.shape)}")
    except Exception as e:
        fail(str(e)); return

    # ── 6  bf16 matmul ───────────────────────────────────────────────────
    step("6. bf16 matmul")
    try:
        a = torch.randn(8, 8, device=device, dtype=torch.bfloat16)
        b = (a @ a.T).cpu()
        ok(f"shape={tuple(b.shape)}")
    except Exception as e:
        fail(str(e)); return

    # ── 7  nn.Linear on MUSA ────────────────────────────────────────────
    step("7. nn.Linear forward+backward bf16")
    try:
        import torch.nn as nn
        m = nn.Linear(64, 64).to(device=device, dtype=torch.bfloat16)
        x = torch.randn(2, 64, device=device, dtype=torch.bfloat16)
        y = m(x)
        y.sum().backward()
        ok(f"grad shape={tuple(m.weight.grad.shape)}")
    except Exception as e:
        fail(str(e)); return

    # ── 8  MultiheadAttention  (need_weights=True → bmm path) ───────────
    step("8. MHA need_weights=True  (bmm path)")
    try:
        mha = nn.MultiheadAttention(64, 4, batch_first=True, dropout=0.0
                                    ).to(device=device, dtype=torch.bfloat16)
        q = torch.randn(2, 8, 64, device=device, dtype=torch.bfloat16)
        out, _ = mha(q, q, q, need_weights=True)
        ok(f"output shape={tuple(out.shape)}")
    except Exception as e:
        fail(str(e))

    # ── 9  MHA need_weights=False  (SDPA path) ──────────────────────────
    step("9. MHA need_weights=False  (SDPA path)")
    try:
        out, _ = mha(q, q, q, need_weights=False)
        ok(f"output shape={tuple(out.shape)}")
    except Exception as e:
        fail(str(e))

    # ── 10  MHA seq_len=1 need_weights=False  (SDPA, known crash shape) ─
    step("10. MHA seq_len=1 need_weights=False  (crash-prone shape)")
    try:
        q1 = torch.randn(2, 1, 64, device=device, dtype=torch.bfloat16)
        out, _ = mha(q1, q1, q1, need_weights=False)
        ok(f"output shape={tuple(out.shape)}")
    except Exception as e:
        fail(str(e))

    # ── 11  TransformerEncoderLayer ──────────────────────────────────────
    step("11. TransformerEncoderLayer  seq_len=8")
    try:
        enc = nn.TransformerEncoderLayer(
            64, 4, 128, batch_first=True
        ).to(device=device, dtype=torch.bfloat16)
        x = torch.randn(2, 8, 64, device=device, dtype=torch.bfloat16)
        y = enc(x)
        ok(f"shape={tuple(y.shape)}")
    except Exception as e:
        fail(str(e))

    # ── 12  TransformerDecoderLayer  tgt seq_len=1 ──────────────────────
    step("12. TransformerDecoderLayer  tgt_len=1")
    try:
        dec = nn.TransformerDecoderLayer(
            64, 4, 128, batch_first=True
        ).to(device=device, dtype=torch.bfloat16)
        tgt = torch.randn(2, 1, 64, device=device, dtype=torch.bfloat16)
        mem = torch.randn(2, 8, 64, device=device, dtype=torch.bfloat16)
        y = dec(tgt, mem)
        ok(f"shape={tuple(y.shape)}")
    except Exception as e:
        fail(str(e))

    # ── 13  F.scaled_dot_product_attention directly ─────────────────────
    step("13. F.scaled_dot_product_attention  seq_len=8")
    try:
        import torch.nn.functional as F
        q = torch.randn(2, 4, 8, 16, device=device, dtype=torch.bfloat16)
        k = torch.randn(2, 4, 8, 16, device=device, dtype=torch.bfloat16)
        v = torch.randn(2, 4, 8, 16, device=device, dtype=torch.bfloat16)
        o = F.scaled_dot_product_attention(q, k, v)
        ok(f"shape={tuple(o.shape)}")
    except Exception as e:
        fail(str(e))

    # ── 14  F.scaled_dot_product_attention  seq_len=1 ───────────────────
    step("14. F.scaled_dot_product_attention  seq_len=1")
    try:
        q1 = torch.randn(2, 4, 1, 16, device=device, dtype=torch.bfloat16)
        k1 = torch.randn(2, 4, 8, 16, device=device, dtype=torch.bfloat16)
        v1 = torch.randn(2, 4, 8, 16, device=device, dtype=torch.bfloat16)
        o = F.scaled_dot_product_attention(q1, k1, v1)
        ok(f"shape={tuple(o.shape)}")
    except Exception as e:
        fail(str(e))

    # ── 15  distributed init ─────────────────────────────────────────────
    if is_distributed:
        step("15. dist.init_process_group  backend=mccl")
        import torch.distributed as dist
        try:
            dist.init_process_group(backend="mccl")
            ok(f"mccl group  rank={dist.get_rank()}/{dist.get_world_size()}")
        except Exception as e:
            fail(f"mccl: {e}")
            log("  trying gloo fallback …")
            try:
                dist.init_process_group(backend="gloo")
                ok("gloo fallback")
            except Exception as e2:
                fail(f"gloo: {e2}"); return

        # ── 16  all_reduce ───────────────────────────────────────────────
        step("16. all_reduce")
        try:
            t = torch.ones(1, device=device)
            dist.all_reduce(t)
            ok(f"result={t.item()}  (expect {world_size})")
        except Exception as e:
            fail(str(e)); return

        # ── 17  all_gather ───────────────────────────────────────────────
        step("17. all_gather")
        try:
            t = torch.tensor([float(_rank)], device=device)
            out = [torch.zeros(1, device=device) for _ in range(world_size)]
            dist.all_gather(out, t)
            ok(f"gathered={[o.item() for o in out]}")
        except Exception as e:
            fail(str(e)); return

        # ── 18  barrier ──────────────────────────────────────────────────
        step("18. barrier")
        try:
            dist.barrier()
            ok()
        except Exception as e:
            fail(str(e)); return
    else:
        log("(steps 15-18 skipped — single GPU)")

    # ── 19  small FSDP wrap + fwd + bwd ──────────────────────────────────
    if is_distributed:
        step("19. FSDP small model  wrap")
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                MixedPrecision,
                ShardingStrategy,
            )
            small = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
            bf16 = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            small = FSDP(small, mixed_precision=bf16,
                         sharding_strategy=ShardingStrategy.FULL_SHARD,
                         device_id=local_rank, use_orig_params=True)
            ok("wrapped")
        except Exception as e:
            fail(str(e)); return

        step("20. FSDP small model  forward")
        try:
            x = torch.randn(2, 64, device=device, dtype=torch.bfloat16)
            y = small(x)
            ok(f"shape={tuple(y.shape)}")
        except Exception as e:
            fail(str(e)); return

        step("21. FSDP small model  backward")
        try:
            y.sum().backward()
            ok()
        except Exception as e:
            fail(str(e)); return
    else:
        log("(steps 19-21 skipped — single GPU)")

    # ── 22  import GPR model ─────────────────────────────────────────────
    step("22. import GPR + GPRConfig")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from config import GPRConfig
        from model import GPR, HSDBlock
        ok()
    except Exception as e:
        fail(str(e)); return

    # ── 23  create GPR model ─────────────────────────────────────────────
    step("23. create GPR model")
    try:
        cfg = GPRConfig().sync()
        gpr = GPR(cfg.model)
        n_p = sum(p.numel() for p in gpr.parameters())
        ok(f"{n_p:,} params  ({n_p/1e6:.1f}M)")
    except Exception as e:
        fail(str(e)); return

    # ── 24  move GPR to MUSA (no FSDP) ───────────────────────────────────
    step("24. GPR → MUSA bf16  (no FSDP)")
    try:
        gpr_plain = GPR(cfg.model).to(device=device, dtype=torch.bfloat16)
        ok()
    except Exception as e:
        fail(str(e)); return

    # ── 25  GPR forward  (single GPU, no FSDP) ──────────────────────────
    step("25. GPR forward  MTP  (single GPU)")
    try:
        B, L = 2, 10
        mc = cfg.model
        batch = {
            "semantic_ids": torch.randint(1, mc.codebook_size, (B, L, mc.n_semantic_levels), device=device),
            "token_types":  torch.randint(0, mc.n_token_types, (B, L), device=device),
            "user_features": torch.randn(B, mc.d_user, device=device, dtype=torch.bfloat16),
            "env_features":  torch.randn(B, mc.d_env, device=device, dtype=torch.bfloat16),
            "seq_len":       torch.tensor([L, L-2], device=device),
            "target_ids":    torch.randint(1, mc.codebook_size, (B, mc.n_semantic_levels), device=device),
        }
        with torch.no_grad():
            result = gpr_plain(batch, mode="mtp")
        ok(f"keys={list(result.keys())}")
    except Exception as e:
        import traceback; traceback.print_exc()
        fail(str(e)); return

    # ── 26  GPR backward  (single GPU) ───────────────────────────────────
    step("26. GPR backward  MTP  (single GPU)")
    try:
        result = gpr_plain(batch, mode="mtp")
        loss = result["all_logits"][0].sum() + result["refine_loss"]
        loss.backward()
        ok(f"loss={loss.item():.4f}")
    except Exception as e:
        import traceback; traceback.print_exc()
        fail(str(e)); return

    del gpr_plain
    torch.musa.empty_cache()

    # ── 27-29  FSDP wrap GPR ─────────────────────────────────────────────
    if is_distributed:
        step("27. FSDP wrap GPR")
        try:
            import functools
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            gpr_fsdp_raw = GPR(cfg.model)
            wrap_pol = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={HSDBlock},
            )
            bf16 = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            gpr_fsdp = FSDP(gpr_fsdp_raw, auto_wrap_policy=wrap_pol,
                            mixed_precision=bf16,
                            sharding_strategy=ShardingStrategy.FULL_SHARD,
                            device_id=local_rank,
                            limit_all_gathers=True,
                            use_orig_params=True)
            ok("wrapped")
        except Exception as e:
            import traceback; traceback.print_exc()
            fail(str(e)); return

        step("28. GPR FSDP forward  MTP")
        try:
            dist.barrier()
            result = gpr_fsdp(batch, mode="mtp")
            ok(f"keys={list(result.keys())}")
        except Exception as e:
            import traceback; traceback.print_exc()
            fail(str(e)); return

        step("29. GPR FSDP backward  MTP")
        try:
            loss = result["all_logits"][0].sum() + result["refine_loss"]
            loss.backward()
            ok(f"loss={loss.item():.4f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            fail(str(e)); return
    else:
        log("(steps 27-29 skipped — single GPU)")

    # ── done ─────────────────────────────────────────────────────────────
    log("=" * 50)
    log("ALL STEPS PASSED")
    log("=" * 50)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"[rank{_rank}] UNHANDLED ERROR:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr, flush=True)
        raise
