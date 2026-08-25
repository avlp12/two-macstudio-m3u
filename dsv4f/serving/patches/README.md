# oMLX overlay patches (3 files)

These are **full replacement modules** for three files inside the installed `omlx`
package, not diffs. `apply.sh` drops them in place (keeping a dated `.orig` backup)
and verifies against `SHA256SUMS`. The three files are byte-identical on both of
our boxes — that is what the checksums are for.

Base: `omlx==0.6.3rc2` (see the reproducibility gap note in the [repo README](../../../README.md#the-honest-reproducibility-gap)).
Licensing: the two `mlx_lm_mtp` files carry `SPDX-License-Identifier: Apache-2.0`;
the `deepseek_v4_model.py` carries an Apple copyright header. They are redistributed
here unmodified except for the changes described below, which are ours.

| File | Target path under `site-packages/` | What it fixes |
|---|---|---|
| `deepseek_v4_model.py` | `omlx/patches/deepseek_v4/deepseek_v4_model.py` | wsdpa fused attention returns wrong results in the MTP/speculative context. Our change gates that path behind `OMLX_WSDPA_MTP=1` (i.e. **off by default**) so the MTP chain falls back to the correct path. Without this you get a live hang / wrong drafts under MTP. |
| `batch_generator.py` | `omlx/patches/mlx_lm_mtp/batch_generator.py` | Fixed-depth MTP dispatch. The stock adaptive depth controller makes its decision per rank, and the two ranks diverge → the collective deadlocks. `OMLX_MTP_FIXED_DEPTH=1` (set in `serve_b.sh`) takes the fixed path. **Never run MTP depth > 1 without this.** |
| `prompt_priming.py` | `omlx/patches/mlx_lm_mtp/prompt_priming.py` | Prompt priming for the MTP head, with the batch-path fix. Folds the prompt into the head's KV cache during prefill instead of starting generation context-starved. Measured on long prompts: conditional accept depth-1 81.5% → 95.6%, depth-3 9.8% → 34.5% (+19.4% decode). |

Upstream: the prompt-priming activation-chain fix was reported as
[jundot/omlx#3079](https://github.com/jundot/omlx/issues/3079).

## Applying

```bash
export BOX_B=<user>@10.0.0.2
./apply.sh                      # this box
scp -r . $BOX_B:/tmp/dsv4f-patches && ssh $BOX_B 'bash /tmp/dsv4f-patches/apply.sh'
```

Both ranks load the same graph. A patch applied on one box only does not produce
an error message — it produces a hang inside a collective, which on the jaccl/RDMA
backend costs you a reboot. Check both.
