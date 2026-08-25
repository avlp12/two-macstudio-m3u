<!-- markdownlint-disable MD013 -->
# Expected results

Numbers measured on our cluster, for you to compare your own run against. Hardware:
2 × Mac Studio M3 Ultra 512 GB, macOS 26.5.2, 3 × TB5 direct (one in the serving path),
`mlx==0.32.0`, `omlx==0.6.3rc2` + the three patches, model
`mlx-community/DeepSeek-V4-Flash-4bit` with the MTP block restored and the round-6c aligned
head. Dates in parentheses are the measurement date.

If your numbers are within a few percent, you have reproduced it. If they are 30% off,
check the two things in [§6](#6-before-you-conclude-anything-is-broken) first.

---

## 1. Interconnect (`cluster/bench.py`, `cluster/jbench.py`)

`all_sum` algorithmic bandwidth = payload bytes / wall time.

| Backend / cables | 4 MB | 32 MB | 128 MB | 512 MB | small-message latency |
|---|---|---|---|---|---|
| `ring` (TCP), 1 cable | 4.15 | 5.10 | 4.61 | 4.51 | 163.6 µs @ 4 KB |
| `ring` (TCP), 3 cables | — | — | — | **9.6** (2.0×) | unchanged |
| `jaccl` mesh (RDMA), 1 cable | — | — | — | **9.31** | **21.2 µs/op @ 8 KB** |
| `jaccl-ring` (RDMA), 3 cables | — | — | — | **15.46** | unchanged |

(GB/s. 2026-08-23.)

Link-level sanity, same wiring: `iperf3` one-way 47.0 / 47.4 / 45.0 Gbit/s at 1 / 4 / 8
streams (stream count is irrelevant — the link is the cap), `--bidir` 39.9 + 40.0 ≈
80 Gbit/s, which is TB5 nominal. MTU 1500 reaches line rate; jumbo frames unnecessary.
`--connections-per-ip` 1/2/4/8 is flat within noise and slightly *worsens* latency.

**`MLX_METAL_FAST_SYNCH` is the single biggest interconnect gotcha:** dependent-chain
`all_sum` on 8 KB, 2-node jaccl — `=0`: **274.8 µs/op**, `=1`: **21.2 µs/op**. A 13×
difference that presents as "TP2 decode is slower than one box".

Adding cables raises bandwidth and never touches latency (177 → 190 µs measured across
cable counts on the ring backend). That is why 3-link serving was rejected: prefill +2.6%,
bs8 aggregate decode −7%.

---

## 2. Prefill

All at **13.9K tokens** unless stated. Prompt is `e1_plans` `n_repeat: 397`, held fixed —
see [§6](#6-before-you-conclude-anything-is-broken).

| Configuration | tok/s | Notes |
|---|---|---|
| 1 box, raw chunk loop (no `BatchGenerator`) | **605.4** | the model's real single-box speed |
| 1 box, via `BatchGenerator` | 395 | the 1.53× harness tax, see below |
| 1 box, E1 matched baseline | 397.4–398.6 | matched stack/length, the honest 1-box baseline |
| 1 box, real-content prompt | 395.1 @13.9K · 383.0 @17.2K | content-control; 1–3% from repeated-doc |
| 1 box, 4K prompt | 470.7 | for reference; oMLX leaderboard entry is 459.9 |
| TP2 jaccl, E1 matched | **574.2–575.1** | **1.44×** true scaling vs the matched 1-box baseline |
| TP2 `ring`, raw chunk loop | 519.6 | ring is *slower* than one box — use jaccl |
| **TP2 serving e2e (`serve_b.sh`)** | **544** | TTFT 25.52 s (ledger range 544–558) |
| **PP2 serving e2e (`serve_b_pp2.sh`)** | **895** | TTFT **15.52 s** = **1.64×**, byte-identical output |
| PP2 stage alone (in the serving process) | 985 | KV handover back to TP2 layout: 126 ms (0.5% of TTFT) |
| PP2 raw, split 22, chunk 2048 | 992.3 | **1.639×** of 1-box raw |
| PP2 raw, split 22, chunk 1024 | 1030.5 | 1.702× — but **different argmax output**, do not ship |

(2026-08-25.)

Chunk-size sweep, TP2, same prompt: **2048 = 426 @4096 (−26%) = 339 @8192 (−41%)**. Bigger
chunks are worse, not better. 2048 is a sharp peak, not a plateau.

Two-box prefill gain is **not** smooth in prompt length. Pooled across repeats:
3.5K ≈ 1.0×, **7K ≈ 0.95× (TP2 slower than one box, 3 of 4 repeats)**, 13.9K 1.44×. Short
lengths also jitter ±20–37% run to run. The transition mechanism is unexplained; do not
extrapolate a smooth curve through it, and do not benchmark TP2 prefill below ~10K and
conclude anything.

Where the two-box ceiling actually is: the bottleneck is **replication, not communication**
— 20.9% of prefill FLOPs are duplicated across ranks under the current sharding (`wo_b`
alone is 16.3%), while communication is 6.5%. Physical ceiling 2.0× (910–1170 tok/s);
current-sharding design ceiling 1.57–1.60×.

### The BatchGenerator prefill tax (worth knowing even if you serve something else)

`BatchGenerator._merge_caches` converts every cache to its `Batch*` variant **even at batch
size 1**. That makes `offset` an `mx.array` instead of an `int`, and all three fused
DeepSeek-V4 prefill kernels bail on their first-line guard — **41 of 43 layers silently fall
back to stock paths**. Ablation: fused-off raw 390.5 ≈ BatchGen 390.6, i.e. 100% of the gap
is the kernel bail, and zero of it is structural overhead. Kernel counters confirm it
(stock: 161 bails / 0 topk; patched: 0 bails / 126 topk).

A B=1 bypass (`bench/b1_cache_bypass.py`) recovers +54.3% on one box (391 → 602.5; with MTP
depth-3, 377.8 → 588.6) with decode completely unaffected — **but it hangs TP2 in lockstep**
under `BatchGenerator` (no-log stop at the first prefill chunk, 600 s deadline, root cause
unidentified). It is shipped **disabled** (`DSV4_B1_CACHE_BYPASS=0` killswitch) and is
included here for the single-box case and for anyone who wants to finish the diagnosis.
PP2 avoids the whole problem structurally by not going through `BatchGenerator` at all.

---

## 3. Decode

| Configuration | tok/s |
|---|---|
| **Production: single stream, MTP depth-3, aligned head, priming** | **45–47** |
| **Production: batch 8, aggregate** | **112–117** |
| 1 box, plain (no MTP) | 30.0–32.4 |
| 1 box, MTP depth-1 | 39.5 |
| TP2 plain, `MLX_METAL_FAST_SYNCH=1` | 33.5–36.0 |
| TP2 plain, `MLX_METAL_FAST_SYNCH=0` | 20.3 |
| TP2 × MTP depth-1 | 44.28 |

(Production rows 2026-08-24; the lower rows are the earlier build-up, kept because they
show what each lever is worth.)

Decode does **not** benefit from extra cables — it is latency-bound and latency is flat in
cable count. TP2 decode only beats a single box once `FAST_SYNCH=1`; before that fix, TP2
decode was 20.3 vs 30.0 single-box and looked like a fundamental jaccl limitation. It was
an environment variable.

---

## 4. MTP acceptance

Telemetry line format (printed once per rank per finished sequence):

```
MTP[0] finish=stop tokens=N cycles=N tok/cycle=X accept=n/d (p%) depth[d1=n/d,d2=n/d,d3=n/d]
```

**Production regime — batch size 1, 24 topics, paired per topic** (this is the one that
matters; `serve_b.sh` activates MTP only at bs1):

| | depth-1 | depth-2 | depth-3 | tok/cycle | sign test |
|---|---|---|---|---|---|
| round 2 (baseline head) | 78.1% | 59.5% | 24.7% | 2.371 | — |
| **round 6c (production)** | **79.3%** | **61.3%** | **34.3%** | **2.459 (+3.68%)** | 19W-4L-1T, p = 0.0026 |
| round 6e (rejected) | 78.4% | 61.9% | 32.4% | 2.436 (−0.91%) | p = 0.38, indistinguishable |

Non-CS topics gained *more* than CS topics (mean Δtok/cycle +0.106 vs +0.064).

**Batch-8 regime, 8 topics** (requires `OMLX_MTP_ROWWISE_BATCH=1`; production never runs
this — included because it is what we measured first and it is directionally consistent):

| | depth-1 | depth-2 | depth-3 | tok/cycle |
|---|---|---|---|---|
| round 2 | 76.5% | 57.1% | 23.0% | 2.312 |
| round 6c | 74.1% | 60.9% | 31.0% | 2.346 (+1.5%) |

**Live serving probe** during the PP2 A/B, 13.9K prompt: d1 73.3%, tok/cycle 2.60 —
identical on both arms, i.e. the PP2 path's loss of prompt priming cost nothing on that
probe.

**Prompt priming**, measured separately on long prompts: conditional accept depth-1
81.5% → 95.6%, depth-3 9.8% → 34.5%, worth +19.4% decode. This is why short requests are
routed around the PP2 path.

Expect **d1 73–79% depending on context**. A d1 far below 70% means your head and your
serving regime disagree — most likely the head is not the aligned one, or `TP2_MTP_CKPT`
loaded on one rank only.

---

## 5. Prefix snapshot cache

| Metric | Value |
|---|---|
| Multi-turn reuse rate | 95% |
| TTFT reduction, multi-turn | −45 – 61% |
| Quality cost | none — greedy-exact |

Not available on the PP2 path (PP2-inserted prompts skip the snapshot store).

Memory: **≈145 GB/box resident under `serve_b_pp2.sh`**, roughly double the non-PP2
resident set, because the PP2 slice is co-resident with the TP2 shard. A clean shutdown
returns both boxes to single-digit GB wired (ours: 9 GB / 6 GB). Anything else means a leak.

---

## 6. Before you conclude anything is broken

**Fix the prompt.** Prefill tok/s here moves by tens of percent with prompt *content*, not
only length. Use a deterministic prompt builder (the `e1_plans` files do), keep it constant
across arms, and never compare a number measured on one prompt to a number measured on
another. Related: `--long-doc 60` is **~2,120 tokens**, not 13.9K; 13.9K is `--long-doc 397`.

**Fix the chunk size at 2048.** It is the peak *and* chunk size changes the output — c1024
and c2048 produce different argmax tokens, on the single-box control too. Chunk size is not
free here.

Other things that will move your numbers, all measured:

- `mlx==0.32.1` in this venv: **−7.6%**, via a native-extension ABI mismatch that silently falls back off the fast kernels. No error, just slower. Pin 0.32.0. (A macOS 27 upgrade must be paired with mlx 0.32.1+, [mlx#4211](https://github.com/ml-explore/mlx/issues/4211) — move both or neither.)
- Thermal drift across a long sweep: **−2.4 – 2.6%** (measured A/B/A). Not enough to explain a large gap, enough to matter at the margin. Also: benchmarking immediately after idle reads ~2.8% low.
- `mx.set_wired_limit(...)` — if you write your own harness and forget it, a >100 GB model thrashes and you will measure something like 0.27 tok/s instead of 22.9. `mlx_lm`'s high-level wrappers set it; calling `generate_step` directly does not.
- Forgetting `mx.eval` in a microbenchmark produces impossible numbers (we once "measured" 437 TFLOPS).
- **Stochastic wedges are real.** On our worst day: 3 transient Metal errors and 2 collective wedges in ~12 hours of heavy two-box work, all recovered by relaunch or reboot, all on unchanged code. Two of the wedges landed on the first topic of a bs1 sweep that had completed cleanly twice before. Budget for it; make long jobs resumable; never store measurements in `/tmp`.

## 7. Numerics you should expect (not bugs)

- **TP2 output is not bit-identical to single-box.** 98.6% of logit elements differ; argmax agrees. This is K-way partial-GEMM floating-point nonassociativity, the same phenomenon as [arXiv 2511.17826](https://arxiv.org/abs/2511.17826) / Thinking Machines' batch-invariance writeup / [DeepSpeed#7500](https://github.com/deepspeedai/DeepSpeed/issues/7500) / vLLM's batch-invariant-kernel work. It is why the MTP head needs re-aligning on TP2 hidden states.
- **PP2 prefill, by contrast, is bit-exact with single-box** — zero logit mismatch over the last 32 positions across the full vocabulary. Layer-splitting does not change any reduction's shape; tensor-parallelism does.
- The B=1 cache bypass is fp-equivalent, not bit-exact (it swaps the kernel on 41 layers).
- On a knife-edge prompt, TP2 vs PP2 greedy output can diverge within the existing TP2 nondeterminism band; a 4-arm ablation showed the KV-injection machinery itself is exact (identical arms produce identical output) and that PP2 tracks single-box more faithfully than TP2 does.
