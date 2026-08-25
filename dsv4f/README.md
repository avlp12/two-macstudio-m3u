<!-- markdownlint-disable MD013 -->
# DeepSeek-V4-Flash-0731 on 2 × M3 Ultra — reproduction guide

Do [the cluster playbook](../README.md) first (wiring, static TB IPs, mDNS diet, venv,
safety rules, `bench.py` interconnect check). This document assumes your two boxes talk to
each other at ~9.3 GB/s over jaccl and that both venvs are identical.

Target numbers are in **[EXPECTED_RESULTS.md](EXPECTED_RESULTS.md)**. Every command below
is one we actually ran; nothing here is invented for the write-up.

Conventions: **box A** = rank 0 (HTTP front end) = `10.0.0.1`, **box B** = rank 1 =
`10.0.0.2`. `$BOX_B` is box B's ssh target, `$BOX_B_HOME` its home directory.

```bash
export BOX_B=<user>@10.0.0.2
export BOX_B_HOME=/Users/<user>
```

---

## Step 1 — Weights

Two pieces: the 4-bit model pack (144 GB) and the aligned MTP head (143 MB).

### 1a. Base 4-bit pack

We did **not** quantize this ourselves — we used the community MLX conversion:

```bash
mkdir -p ~/dsv4flash
hf download mlx-community/DeepSeek-V4-Flash-4bit --local-dir ~/dsv4flash/mlx4bit
```

(41 files, ~144 GB; ours took ~33 minutes. We pulled it with
`huggingface_hub.snapshot_download`; the `hf` CLI does the same thing.)

The quantization recipe is recorded in the pack's own `config.json`, and it is a
**mixed-precision** layout, not uniform 4-bit — worth knowing if you ever rebuild it:

| Scope | Mode | Bits | Group size | Count |
|---|---|---|---|---|
| default (attention, embeddings, `lm_head`, shared experts, gates) | `affine` | 4 | 64 | 522 per-tensor overrides + the default |
| all routed-expert `switch_mlp.{gate,up,down}_proj`, layers 0–43 | `mxfp4` | 4 | 32 | 132 |

Model shape, for sanity-checking your load: 43 layers, `hidden_size` 4096, 256 routed
experts + 1 shared, 6 experts/token, MLA with `num_key_value_heads: 1` and `head_dim` 512,
`index_topk` 512 (the DSA sparse indexer), `num_nextn_predict_layers: 1` (the MTP block),
`max_position_embeddings` 1,048,576, and a `compress_ratios` list that alternates 4/128 per
layer — that alternation is why the PP2 pipeline split lands at layer 22 and not 21 or 23.

### 1b. Restore the MTP block (required — the community conversion drops it)

`mlx-community/DeepSeek-V4-Flash-4bit` has **no `mtp.*` tensors**. Without them there is no
self-speculative decoding, and decode falls from 45–47 tok/s to the low 30s. We restored
the block from the original (unquantized) DeepSeek-V4-Flash-0731 checkpoint.

Two sub-steps:

1. **Extract** every tensor whose name starts with `mtp.` from the original checkpoint into
   a single `mtp_raw.safetensors` (1,585 tensors here; 3.59 GB). Dtypes present and
   preserved: `BF16`, `F32`, `F8_E4M3`, `F8_E8M0`, `I8`.

   > **Gap, stated plainly:** we did this extraction on a different machine (the rig that
   > held the original checkpoint) and the extraction script was not preserved. It is a
   > mechanical name-prefix filter with no logic worth guarding, and `restore_mtp.py` below
   > documents exactly which dtypes and key names it expects on the input side — but we are
   > not going to pretend we are handing you a script we can no longer produce.

2. **Restore** it into the pack with `weights/restore_mtp.py`, which mirrors the backbone's
   representation choices exactly:

   ```bash
   DSV4_PACK=~/dsv4flash/mlx4bit DSV4_MTP_RAW=~/dsv4flash/mtp_raw.safetensors \
     ~/venv_omlx063/bin/python dsv4f/weights/restore_mtp.py
   # -> "RESTORE-DONE"
   ```

   What it does: routed experts are a **byte transplant** of mxfp4 g32 (no requantization);
   the 10 fp8 tensors are dequantized through an e4m3fn lookup table and requantized to
   affine 4/64; 19 tensors (norms, attention sink, hash-collision params, gate bias) are
   copied raw. It then writes `model-mtp-00001-of-00001.safetensors` (3.52 GB), updates
   `model.safetensors.index.json`, and adds 13 quantization overrides to `config.json`.
   Both files are backed up as `*.premtp` first.

   Two renames it handles that will silently break a hand-rolled version:
   `ffn.gate.bias` → `e_score_correction_bias`, and `ffn.experts.<E>.w{1,3,2}` →
   `ffn.switch_mlp.{gate,up,down}_proj` (the post-stacking path). The `config.json`
   quantization override keys for the MTP block need the nested `.block.` path
   (`mtp.0.block.ffn...`) because of how `MTPBlock` wraps it.

   **Pitfall:** after restoration the stock mlx-lm loader rejects the pack. Keep a
   symlinked view built from the `.premtp` index if you ever need to load it with stock
   tooling (`~/dsv4flash/mlx4bit_nomtp` for us). The oMLX overlay loads the restored pack
   fine — the serving stack is the thing that needs it.

### 1c. Aligned MTP head

The MTP head shipped in the checkpoint is chain-aligned on *single-box* hidden states,
while TP2 serving feeds it *TP2* hidden states — the two differ by K-way partial-GEMM
floating-point nonassociativity, and the head's acceptance rate pays for it. Our round-6c
head is retrained on captured TP2 hidden states and is what production runs:

```bash
mkdir -p ~/dsv4flash/align/ckpt_r6c_real
hf download avlp12/dsv4flash-mtp-aligned mtp_aligned_r6c_step5000.safetensors \
  --local-dir /tmp/mtp-dl
mv /tmp/mtp-dl/mtp_aligned_r6c_step5000.safetensors \
   ~/dsv4flash/align/ckpt_r6c_real/step5000.safetensors
shasum -a 256 ~/dsv4flash/align/ckpt_r6c_real/step5000.safetensors
# 725f300742d3f44dfa33d07990b4334455044a78d71049d18282122a99caf00b   (150,254,980 bytes)
```

The same repo keeps `mtp_aligned_r2_step1000.safetensors` (the round-2 head) for rollback —
put it at `~/dsv4flash/align/ckpt_r2/step1000.safetensors` and `serve_b.sh` has the
one-line rollback commented in. `align/` explains how the head was trained and how to
retrain it; you do **not** need to retrain anything to reproduce the serving numbers.

---

## Step 2 — venv

Per the [root README §3](../README.md#3-software): Python 3.11, `mlx==0.32.0`, the pinned
`mlx-lm` commit, `omlx==0.6.3rc2`, then the three overlay patches:

```bash
bash dsv4f/serving/patches/apply.sh ~/venv_omlx063                 # box A
scp -r dsv4f/serving/patches $BOX_B:/tmp/dsv4f-patches
ssh $BOX_B 'bash /tmp/dsv4f-patches/apply.sh ~/venv_omlx063'       # box B
```

Both must print `PATCH-OK`. Read [`serving/patches/README.md`](serving/patches/README.md)
for what each patch fixes — in particular, `OMLX_MTP_FIXED_DEPTH` is not optional at
depth > 1, and the wsdpa-in-MTP-context path must stay off.

---

## Step 3 — Deploy to both boxes

Everything runs from `/Users/Shared/tp2` on **both** machines. That path is **not shared** —
it is a local directory on each box that happens to have the same name, so you have to
copy files to both, every time.

```bash
sudo mkdir -p /Users/Shared/tp2 && sudo chown $(whoami) /Users/Shared/tp2
cp dsv4f/serving/*.py dsv4f/serving/*.sh dsv4f/bench/*.py dsv4f/bench/*.sh /Users/Shared/tp2/
cp dsv4f/align/dump_hidden_tp2_corpus.py /Users/Shared/tp2/
cp -r dsv4f/bench/e1_plans /Users/Shared/tp2/
mkdir -p /Users/Shared/tp2/exp_chain
for t in jaccl2 jaccl_r3 ring; do
  sed "s|BOX_B_USER@10.0.0.2|$BOX_B|" cluster/hostfile_$t.json.template \
    > /Users/Shared/tp2/hostfile_$t.json
done

rsync -a /Users/Shared/tp2/ $BOX_B:/Users/Shared/tp2/
# the MTP checkpoint has to exist at the same $HOME-relative path on both boxes
rsync -a ~/dsv4flash/align/ckpt_r6c_real/step5000.safetensors \
  $BOX_B:$BOX_B_HOME/dsv4flash/align/ckpt_r6c_real/
```

Then **verify by checksum**, do not assume:

```bash
BOX_B=$BOX_B bash cluster/check_deploy_sync.sh
# -> DEPLOY-SYNC-OK
```

This is not ceremony. `rsync -rt` does not detect silent corruption, and asymmetric
deployment does not produce an error — it produces a **hang inside a collective**, which on
jaccl costs you a reboot. We have hit rank-asymmetric deployment more than once; it is the
single most common way to lose an hour here.

Two related traps:

- `mlx.launch` does **not** propagate your local shell environment to the remote rank. A `${VAR:-default}` inside a launcher changes rank 0 only, and you get a silently asymmetric two-rank config. That is why `serve_b_pp2.sh` hardcodes its env values as literals.
- Both boxes need the model pack at the **same `$HOME`-relative path** (`~/dsv4flash/mlx4bit`), and both need the MTP checkpoint. `serve_b.sh` gate 43 checks the checkpoint locally on each rank precisely because a one-sided miss hangs the partner instead of failing.

---

## Step 4 — Start serving

Both launchers are run **by `mlx.launch`, on both hosts**. You invoke `mlx.launch` once,
from box A.

```bash
cd /Users/Shared/tp2 && nohup ~/venv_omlx063/bin/mlx.launch \
  --hostfile hostfile_jaccl2.json /Users/Shared/tp2/serve_b.sh > serve_b.log 2>&1 &
```

**Judge READY by the "서빙 시작" line in `serve_b.log`, never by port occupancy** — a ghost
single-slot server can hold :8003 with dead workers behind it, and you will spend twenty
minutes benchmarking nothing.

The server is OpenAI-compatible on `:8003`, continuous batching up to 8 slots, with a
prefix-snapshot cache for multi-turn reuse.

### Which launcher: `serve_b.sh` vs `serve_b_pp2.sh`

| | `serve_b.sh` (default) | `serve_b_pp2.sh` (PP2 prefill) |
|---|---|---|
| Prefill @13.9K, e2e | 544 tok/s, TTFT 25.5 s | **895 tok/s, TTFT 15.5 s (1.64×)** |
| Decode | unchanged | unchanged |
| How prefill runs | TP2 tensor-parallel through `BatchGenerator` | layer-split 2-stage pipeline (raw chunk loop, no collectives), KV handed back to the TP2 layout |
| Applies to | every request | only requests ≥ `DSV4_PP2_MIN_TOKENS` (4096); shorter requests keep the legacy path |
| Prefix-snapshot cache | yes | **no** for PP2-inserted prompts — they skip the snapshot store, so you lose multi-turn TTFT reuse on long prompts |
| MTP prompt priming | yes | **no** on the PP2 path — those prompts insert unprimed and fall back (this is why short requests are routed around it) |
| Resident memory | ~72 GB/box | **~145 GB/box** (the PP2 slice is co-resident) — the wired killswitch sidecar is mandatory |
| Output | — | byte-identical long-form response vs. the OFF arm; identical MTP telemetry |

Rule of thumb: PP2 is a straight win for long-prompt, low-multi-turn workloads (document
Q&A, agentic long-context reads). If your traffic is chatty multi-turn with long shared
prefixes, the snapshot cache the legacy path keeps (95% reuse, TTFT −45–61%) may be worth
more than 1.64× on the cold prefill. Measure your own mix; both launchers are here.

**Chunk size is fixed at 2048 and is not output-invariant.** `DSV4_PP2_CHUNK=1024` is
measurably faster raw (1030 vs 992 tok/s) but produces *different argmax output* — the same
is true on the single-box control, so it is not a PP2 bug, it is a property of chunked
prefill here. Larger chunks are worse *and* wrong: 4096 measured −26%, 8192 −41%. Keep 2048.

### Stopping

Use `cluster/tp2_guard.sh`'s `tp2_safe_shutdown` (quiesce 20 s, TERM, wait 3 min, never
KILL). After it returns, confirm both boxes dropped back to low wired memory before
launching anything else — a clean stop looks like ~9 GB / ~6 GB, not 90 GB.

---

## Step 5 — Verify

### 5a. Long-context integrity (needle) — do this before you trust any number

There is a public report of the jaccl backend corrupting the DSA indexer across multiple
prefill chunks. It **does not reproduce on this stack**, but you should confirm that on
your own hardware before you believe your throughput numbers, because a corrupted indexer
is fast and wrong.

```bash
# run under mlx.launch on both boxes (TP2), and again with --require-world 1 (single box)
cd /Users/Shared/tp2 && ~/venv_omlx063/bin/mlx.launch \
  --hostfile hostfile_jaccl2.json /Users/Shared/tp2/e1_needle_worker.sh
```

`e1_plans/needle.json` hides a 4-digit `SECRET-CODE` at 50% depth in prompts of 13.9K /
2.2K / 1.8K tokens and asks for it back. Pass = the exact code recovered at all three
lengths on both 1-box and TP2, with the surrounding reasoning text matching verbatim. Ours
did; total runtime ~7 minutes.

### 5b. Prefill throughput

```bash
# TP2
cd /Users/Shared/tp2 && ~/venv_omlx063/bin/mlx.launch \
  --hostfile hostfile_jaccl2.json /Users/Shared/tp2/e1_timing_worker.sh
# single box, with a hard 600 s deadline and TERM-only cleanup
bash /Users/Shared/tp2/pa12_run.sh ~/venv_omlx063/bin/python \
  /Users/Shared/tp2/e1_plans/timing_1box.json ~/dsv4flash/align/logs/e1_1box.log
```

Results print as one `[E1-RESULT] {json}` line per step — grep the log for them. The model
is loaded **once** and every step in the `--plan` runs against it, deliberately: each TP2
launch is a jaccl-mesh wedge risk, so minimize launches.

> **Fix your prompt before you compare anything.** Prefill tok/s on this model moves by
> tens of percent with prompt *content*, not just length. Our plans build a deterministic
> repeated-document prompt (`n_repeat: 397` ≈ 13.9K tokens); `e1_plans/pa2_content.json`
> is the real-content control we used to check that the repeated document was not
> pathological (it was not — 1–3% apart). Compare like-for-like or the numbers are noise.
> Also: `--long-doc 60` is ~2,120 tokens, **not** 13.9K. 13.9K is `--long-doc 397`.

### 5c. Serving-path prefill A/B (the 895 number)

Start with `serve_b.sh`, send a 13.9K-token prompt, record TTFT and e2e prefill tok/s from
the server log. Shut down cleanly. Start with `serve_b_pp2.sh`, send the **identical**
prompt, compare. You should see 544 → 895 tok/s, TTFT 25.5 → 15.5 s, and a byte-identical
response. The PP2 stage itself runs at ~985 tok/s; the KV handover back to the serving
layout costs 126 ms (0.5% of TTFT).

### 5d. Decode and MTP telemetry

```bash
cd /Users/Shared/tp2 && ~/venv_omlx063/bin/mlx.launch \
  --hostfile hostfile_jaccl2.json /Users/Shared/tp2/rt.sh
# depth comes from /Users/Shared/tp2/exp_chain/depth.cfg (default 1); production is 3
```

The overlay prints one telemetry line per finished sequence:

```
MTP[0] finish=stop tokens=512 cycles=208 tok/cycle=2.462 accept=304/416 (73.1%) depth[d1=163/208,d2=97/159,d3=44/142]
```

How to read it:

- **`tok/cycle`** is the number that matters — committed tokens per verify cycle. 1.0 means MTP is doing nothing for you. Production sits at **2.35–2.46**.
- **`d1/d2/d3`** are per-depth acceptance: accepted / offered at that chain position. `d1` 73–79%, `d2` ~61%, `d3` ~31–34% is healthy. Note `d2`'s denominator is smaller than `d1`'s — depth *n* is only offered when depth *n−1* was accepted.
- The line is printed **once per rank**, so you will see each result twice under TP2. Take the first.
- **MTP auto-disables at batch > 1.** `serve_b.sh` has no rowwise override, so production activates MTP only at batch size 1. If you want to measure MTP at bs8 you must set `OMLX_MTP_ROWWISE_BATCH=1` — and be aware that this is a regime production never runs in, which is exactly the mistake an external audit caught us making.

`align/p1_paired_analysis.py` parses these lines out of a `run_tp2_flash.py --all-topics`
log and does the per-topic pairing, sign test, and pooled aggregate — that is how the
round-2 vs round-6c comparison in [EXPECTED_RESULTS.md](EXPECTED_RESULTS.md) was produced.

---

## Step 6 (optional) — Re-align the MTP head yourself

Not needed to reproduce serving. Here because the recipe generalizes: **if your speculative
head was trained on hidden states from a different execution regime than the one you serve
in, it will underperform, and synthetic noise will not fix it.**

We spent three failed rounds on that lesson (LoRA on the quantized shared experts; Gaussian
noise at 1% and at 0.3% of hidden std, meant to imitate the drift). All three improved
offline eval and *regressed live acceptance*. The failure writeup, including four separate
VJP blockers you hit trying to fine-tune through a quantized MoE forward, is in
[alis-dwq PORTING_INTEGRITY.md §12](https://github.com/avlp12/alis-dwq/blob/main/docs/PORTING_INTEGRITY.md#12-fine-tuning-through-a-quantized-moe-forward-four-vjp-blockers-in-the-order-you-hit-them-and-why-a-working-gradient-still-isnt-a-working-result).

What worked (round 6c):

```bash
# 1. on-policy corpus — greedy self-continuation, single box, ~660 generations x 400 tok.
#    Ours is included: align/corpus/onpolicy_c.txt (1.16 MB). To regenerate:
~/venv_omlx063/bin/python ~/dsv4flash/align/gen_onpolicy_v3.py

#    It reads the file list in align/corpus/corpus_onpolicy_c_portable.txt, which points at
#    onpolicy_c.txt — put both under ~/dsv4flash/align/ before running the capture.

# 2. capture REAL TP2 hidden states (forward-only, no gradient — safe under R5 on ring)
cd /Users/Shared/tp2 && ~/venv_omlx063/bin/mlx.launch --hostfile hostfile_ring.json \
  -- ~/venv_omlx063/bin/python /Users/Shared/tp2/dump_hidden_tp2_corpus.py
#    697 windows of 384 tokens, ~82 min, resumable per window.
#    Writes exp_chain/r6c_real_hidden{.safetensors,_ids.json} on each box; copy the pair
#    to ~/dsv4flash/align/ (as r6c_real_hidden*) before training.

# 3. train the non-expert part of mtp.0 (74.3M params, bf16) on those hiddens — single box
cd ~/dsv4flash/align && ~/venv_omlx063/bin/python -u train_align.py --steps 5000 --lr 5e-6 \
  --real-hidden r6c_real_hidden --init-ckpt ckpt_r2/step1000.safetensors \
  --out ckpt_r6c_real
```

Notes that cost us time:

- The experts stay frozen (mxfp4, `stop_grad`); the HC custom kernel falls back on a train gate; loss is `CE_d2 + 0.3 * CE_d1`.
- The wsdpa fused prefill kernel **live-hangs** (GPU command-buffer timeout) when called repeatedly on 384-token corpus windows. Both `train_align.py` and `dump_hidden_tp2_corpus.py` disable it and fall back to stock SDPA. A single 505-token forward does not reproduce it; the repeated-call pattern does.
- Design long captures to be resumable **per window** (one file per window, skip what exists). Ours survived exactly the stochastic wedge that rule exists for.
- `mx.save_safetensors` appends the extension to a path that lacks one — watch your atomic-write temp filenames.
- Validate with **bs1 × many topics**, paired, with a sign test. Our bs8 × 8-topic result was directionally right but measured a regime production never enters, and the topic set skewed toward the training corpus. The bs1 × 24-topic redo (16 non-CS topics added) settled it: +3.68% tok/cycle, 19W-4L-1T, p = 0.0026, with non-CS topics gaining *more* than CS ones.

A round-6d LoRA add-on over the same real-hidden recipe regressed back to baseline and
closed that line. A round-6e "serving-faithful" capture (bulk prefill + teacher-forced
decode) raised offline eval by +7 pp and moved live acceptance not at all — the residual
offline↔live gap is in the live loop's conditional structure (acceptance-dependent draft
position distribution, chain input, priming history), not in the hidden-state capture
regime. We stopped there.
