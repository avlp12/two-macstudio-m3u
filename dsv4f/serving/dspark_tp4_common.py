"""Shared load, shard and MTP-log helpers for Pro-0813 TP4 + DSpark."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import mlx.core as mx
from mlx.nn.layers.distributed import shard_inplace, shard_linear

DEFAULT_MODEL = "/opt/models/DeepSeek-V4-Pro-0813-MXFP4-MLX"


def consensus(value: mx.array, group, label: str) -> None:
    gathered = mx.distributed.all_gather(value, group=group).reshape(group.size(), -1)
    mx.eval(gathered)
    if not bool(mx.all(gathered == gathered[0]).item()):
        raise RuntimeError(f"rank disagreement after {label}: {gathered.tolist()}")


def shard_mtp(model, group) -> int:
    stages = getattr(model, "mtp", None)
    if not stages:
        raise RuntimeError("model has no mtp modules; DSpark head was not attached")
    world = group.size()
    rank = group.rank()
    o_groups = model.args.o_groups
    for stage in stages:
        stage = getattr(stage, "block", stage)  # Flash legacy MTPBlock 은 .block 중첩
        attn = stage.attn
        attn.sharding_group = group
        attn.wq_b = shard_linear(
            attn.wq_b,
            "all-to-sharded",
            segments=o_groups,
            group=group,
        )
        shard_inplace(attn.wo_a, "sharded-to-all", group=group)
        attn.attn_sink = mx.split(attn.attn_sink, world)[rank]
        attn.n_heads //= world

        ffn = stage.ffn
        ffn.sharding_group = group
        shard_inplace(ffn.shared_experts.gate_proj, "all-to-sharded", group=group)
        shard_inplace(ffn.shared_experts.down_proj, "sharded-to-all", group=group)
        shard_inplace(ffn.shared_experts.up_proj, "all-to-sharded", group=group)
        shard_inplace(ffn.switch_mlp.gate_proj, "all-to-sharded", group=group)
        shard_inplace(ffn.switch_mlp.down_proj, "sharded-to-all", group=group)
        shard_inplace(ffn.switch_mlp.up_proj, "all-to-sharded", group=group)
    return len(stages)


def materialize(model, rank: int, group) -> None:
    heads = []
    inner = getattr(model, "model", model)
    for name in ("embed_tokens", "norm", "hc_head"):
        module = getattr(inner, name, None)
        if module is not None:
            heads.append(module)
    if getattr(model, "lm_head", None) is not None:
        heads.append(model.lm_head)
    if heads:
        mx.eval(*[module.parameters() for module in heads])
    consensus(mx.array([0], dtype=mx.uint32), group, "non-layer weights")

    layers = model.model.layers
    for index, layer in enumerate(layers):
        started = time.monotonic()
        mx.eval(layer.parameters())
        mx.synchronize()
        consensus(
            mx.array([index + 1], dtype=mx.uint32),
            group,
            f"layer {index + 1}",
        )
        print(
            f"[pro0813-dspark-load] rank={rank} layer={index + 1}/{len(layers)} "
            f"seconds={time.monotonic() - started:.2f}",
            flush=True,
        )

    stages = list(model.mtp)
    for index, stage in enumerate(stages):
        started = time.monotonic()
        mx.eval(stage.parameters())
        mx.synchronize()
        consensus(
            mx.array([1000 + index], dtype=mx.uint32),
            group,
            f"mtp {index}",
        )
        print(
            f"[pro0813-dspark-load] rank={rank} mtp={index + 1}/{len(stages)} "
            f"seconds={time.monotonic() - started:.2f}",
            flush=True,
        )


class MtpLogTap(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("MTP path activated") or message.startswith("MTP["):
            self.lines.append(message)


def snapshot_mtp(batch_gen) -> dict[str, Any]:
    gen_batch = getattr(batch_gen, "_generation_batch", None)
    state = getattr(gen_batch, "_omlx_mtp_state", None) if gen_batch is not None else None
    if state is None or getattr(state, "stats", None) is None:
        return {}
    stats = state.stats
    drafted = sum(getattr(stats, "depth_drafted", []) or [])
    accepted = int(getattr(stats, "accepts", 0) or 0)
    cycles = int(getattr(stats, "cycles", 0) or 0)
    emits = (
        int(getattr(stats, "init_emits", 0) or 0)
        + int(getattr(stats, "draft_emits", 0) or 0)
        + int(getattr(stats, "bonus_emits", 0) or 0)
        + int(getattr(stats, "verify_emits", 0) or 0)
    )
    return {
        "cycles": cycles,
        "accepts": accepted,
        "drafted": int(drafted),
        "accept_pct": (100.0 * accepted / drafted) if drafted else 0.0,
        "tok_per_cycle": (emits / cycles) if cycles else 0.0,
        "emits": emits,
        "init": int(getattr(stats, "init_emits", 0) or 0),
        "draft": int(getattr(stats, "draft_emits", 0) or 0),
        "bonus": int(getattr(stats, "bonus_emits", 0) or 0),
        "verify": int(getattr(stats, "verify_emits", 0) or 0),
        "backbone_ms": float(getattr(stats, "backbone_ms", 0.0) or 0.0),
        "mtp_ms": float(getattr(stats, "mtp_head_ms", 0.0) or 0.0),
    }


_RING_STATE: dict[str, Any] = {"mode": None, "orig_has": None, "orig_gemm": None}


def configure_ring_gemm(mode: str, *, heads: int, rank: int = 0) -> str:
    """Select the TP4 ring-GEMM strategy.

    off    — hide ``dspark_ring_gemm``; verify uses the gather fallback.
    pad    — keep the stock 64-head kernel; pad 32-head lhs to 64 and slice.
    native — call the kernel as-is (for a 32-head dylib in a side venv).
    """
    from omlx.custom_kernels.glm_moe_dsa import fast

    mode = (mode or "off").strip().lower()
    if mode not in {"off", "pad", "native"}:
        raise ValueError(f"unknown ring mode {mode!r}")
    if _RING_STATE["orig_has"] is None:
        _RING_STATE["orig_has"] = fast.has_symbol
        _RING_STATE["orig_gemm"] = fast.dspark_ring_gemm

    orig_has = _RING_STATE["orig_has"]
    orig_gemm = _RING_STATE["orig_gemm"]

    def has_off(name: str, _orig=orig_has) -> bool:
        if name == "dspark_ring_gemm":
            return False
        return _orig(name)

    def gemm_pad(lhs, source, indices, transpose_rhs, **kwargs):
        n_heads = int(lhs.shape[1])
        if n_heads == 64:
            return orig_gemm(lhs, source, indices, transpose_rhs, **kwargs)
        if n_heads != 32:
            raise ValueError(
                f"pad ring GEMM expects 32 or 64 heads, got {n_heads}"
            )
        pad = mx.zeros((lhs.shape[0], 64 - n_heads, lhs.shape[2]), dtype=lhs.dtype)
        lhs64 = mx.contiguous(mx.concatenate([lhs, pad], axis=1))
        out = orig_gemm(lhs64, source, indices, transpose_rhs, **kwargs)
        return out[:, :n_heads, :]

    if mode == "off" or heads == 64 and mode != "native":
        fast.has_symbol = has_off if heads != 64 else orig_has
        fast.dspark_ring_gemm = orig_gemm
        applied = "off" if heads != 64 else "stock64"
    elif mode == "pad":
        fast.has_symbol = orig_has
        fast.dspark_ring_gemm = gemm_pad
        applied = "pad"
    else:
        fast.has_symbol = orig_has
        fast.dspark_ring_gemm = orig_gemm
        applied = "native"

    _RING_STATE["mode"] = applied
    if rank == 0:
        print(
            f"[pro0813-dspark] ring_gemm mode={applied} rank_heads={heads}",
            flush=True,
        )
    return applied


def load_sharded_dspark(model_path: str, group, depth: int = 5):
    """Apply overlays, load lazy, shard backbone + MTP, materialise."""
    os.environ.setdefault("MLX_METAL_FAST_SYNCH", "0")

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
    from omlx.patches.mlx_lm_mtp import (
        apply_mlx_lm_mtp_patch,
        set_mtp_active,
        set_mtp_depth,
    )
    from mlx_lm import load

    apply_deepseek_v4_patch()
    if not apply_mlx_lm_mtp_patch():
        raise RuntimeError("apply_mlx_lm_mtp_patch() refused; DSpark overlay missing")
    set_mtp_active(True)
    set_mtp_depth(depth)

    rank = group.rank()
    import mlx.core as _mx
    _mx.set_wired_limit(_mx.metal.device_info()["max_recommended_working_set_size"])
    t0 = time.monotonic()
    model, tokenizer = load(model_path, lazy=True)
    if not hasattr(model, "shard"):
        raise RuntimeError("loaded model has no shard(); omlx base patch did not apply")
    if not getattr(model, "_omlx_dspark_decode_enabled", False):
        if getattr(model, "_omlx_mtp_decode_enabled", False):
            pass  # Flash 0731: legacy MTP-1 (DSpark 블록 없음) — 정상 경로
        else:
            raise RuntimeError("MTP/DSpark decode flag is off after load")
    if not getattr(model, "mtp", None):
        raise RuntimeError("sanitize dropped mtp.*")
    model._omlx_mtp_depth = min(
        int(getattr(model, "_omlx_mtp_depth", depth) or depth),
        int(getattr(model.args, "dspark_block_size", depth) or depth),
    )
    model.shard(group)
    n_mtp = shard_mtp(model, group)
    heads = int(model.model.layers[0].attn.n_heads)
    configure_ring_gemm(os.environ.get("PRO0813_DSPARK_RING", "off"), heads=heads, rank=rank)
    materialize(model, rank, group)
    load_s = time.monotonic() - t0
    if rank == 0:
        print(
            f"[pro0813-dspark] load+shard seconds={load_s:.1f} "
            f"layers={len(model.model.layers)} mtp={n_mtp} "
            f"depth={model._omlx_mtp_depth} "
            f"targets={list(getattr(model.args, 'dspark_target_layer_ids', []) or [])} "
            f"heads={model.model.layers[0].attn.n_heads} "
            f"mtp_heads={getattr(getattr((model.mtp or [None])[0], chr(97)+chr(116)+chr(116)+chr(110), None), chr(110)+chr(95)+chr(104)+chr(101)+chr(97)+chr(100)+chr(115), '?')}",
            flush=True,
        )
    return model, tokenizer, load_s
