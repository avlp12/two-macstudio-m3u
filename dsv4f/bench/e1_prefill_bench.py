"""E1 — TP2 프리필 매칭 기준선 + 청크 스윕 + needle 정합성 하네스.

PREFILL_CEILING_INVESTIGATION_2026-08-25.md 의 E1 설계를 그대로 실행하기 위한
run_tp2_flash.py 각색판. 모델은 **한 번만 로드**하고 --plan 에 나열된 여러 측정을
순차 실행한다(TP2 발사 횟수를 최소화 — 매 발사가 jaccl 메시 웨지 위험이므로).

--plan 은 JSON 배열 파일, 각 원소:
  {"mode": "needle"|"timing", "label": "13.9K", "n_repeat": 397,
   "chunk": 2048, "reps": 1, "max_tokens": 16,
   "secret_code": "8317", "needle_frac": 0.5, "warmup": false, "tag": "..."}

결과는 rank0 에서 스텝마다 `[E1-RESULT] {json}` 한 줄로 stdout 에 찍는다(로그 grep용).
"""
import argparse, json, os, sys, time
import logging
logging.basicConfig(level=logging.INFO)

os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
import mlx.core as mx

sys.path.insert(0, "/Users/Shared/tp2")

FILLER = "분산 텐서 병렬과 압축 어텐션의 상호작용은 위치 기하와 수용률 경제학에 민감하다. "
QUESTION = ("위 참고 문서 안에 있던 SECRET-CODE 대괄호 안의 숫자가 무엇입니까? "
            "숫자만 답하세요.")


def build_plain(n_repeat: int, tail: str) -> str:
    body = ("참고 문서: " + FILLER * n_repeat + " ") if n_repeat else ""
    return body + tail


def build_file_body(path: str, target_tokens: int, tok, char_start: int = 0,
                     char_len: int | None = None, tol: int = 30, max_iter: int = 6):
    """실제 콘텐츠 파일에서 target_tokens 에 근접하는 접두 슬라이스를 찾는다
    (PA2 — 반복-문서 병리성 대조용). char_len 이 명시되면 탐색 없이 그대로 슬라이스한다
    (재현 시 정확 재현용). 반환: (body_text, recipe_dict)."""
    with open(path, "r", encoding="utf-8") as f:
        full_text = f.read()

    if char_len is not None:
        body_text = full_text[char_start:char_start + char_len]
        n_tok = len(tok.encode(body_text))
        recipe = {"file": path, "char_start": char_start, "char_len": char_len,
                  "body_tokens": n_tok, "search": "fixed"}
        return body_text, recipe

    # 적응 탐색: chars/token 비 추정 후 반복 보정
    ratio = 3.0
    guess = max(64, int(target_tokens * ratio))
    n_tok = 0
    for _ in range(max_iter):
        guess = min(guess, len(full_text) - char_start)
        body_text = full_text[char_start:char_start + guess]
        n_tok = len(tok.encode(body_text))
        if abs(n_tok - target_tokens) <= tol or guess >= len(full_text) - char_start:
            break
        ratio = guess / max(n_tok, 1)
        guess = max(64, int(target_tokens * ratio))
    recipe = {"file": path, "char_start": char_start, "char_len": guess,
              "body_tokens": n_tok, "target_tokens": target_tokens, "search": "adaptive"}
    return body_text, recipe


def build_needle(n_repeat: int, code: str, frac: float, question: str) -> str:
    reps = [FILLER] * n_repeat
    pos = max(1, int(n_repeat * frac))
    needle_sentence = f"[[SECRET-CODE:{code}]] 이 값을 정확히 기억하세요. "
    reps.insert(pos, needle_sentence)
    body = "참고 문서: " + "".join(reps) + " "
    return body + question


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/dsv4flash/mlx4bit"))
    ap.add_argument("--require-world", type=int, default=2)
    ap.add_argument("--plan", required=True, help="측정 스텝 목록 JSON 파일 경로")
    args = ap.parse_args()

    with open(args.plan) as f:
        plan = json.load(f)

    group = mx.distributed.init()
    rank, world = group.rank(), group.size()
    print(f"[rank {rank}] world={world}", flush=True)
    assert world == args.require_world, f"world {world} != {args.require_world}"

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
    apply_deepseek_v4_patch()
    from mlx_lm import load
    from dspark_tp4_common import shard_mtp

    t0 = time.monotonic()
    model, tok = load(args.model, lazy=True)
    assert hasattr(model, "shard"), "omlx base patch 미적용"
    if world > 1:
        model.shard(group)
        try:
            shard_mtp(model, group)
        except Exception as e:
            if rank == 0:
                print(f"[e1] mtp 샤딩 생략: {e}", flush=True)
    info = mx.metal.device_info()
    mx.set_wired_limit(info["max_recommended_working_set_size"])
    for layer in model.model.layers:
        mx.eval(layer.parameters())
        mx.synchronize()
    mx.eval(model.parameters())
    mx.synchronize()
    if rank == 0:
        print(f"[e1] load+shard {time.monotonic()-t0:.1f}s world={world}", flush=True)

    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    mx.random.seed(7)  # 전 랭크 동일 시드 (JACCL 요건)

    def run_once(ids, max_tokens, chunk):
        t_submit = time.perf_counter()
        bg = BatchGenerator(model, max_tokens=max_tokens, sampler=make_sampler(0.0),
                             completion_batch_size=1, prefill_batch_size=1,
                             prefill_step_size=chunk)
        bg.insert([ids], max_tokens=[max_tokens])
        t_first = None
        toks = []
        done = 0
        while done < 1:
            rs = bg.next_generated()
            if not rs:
                break
            if t_first is None:
                t_first = time.perf_counter()
            for r in rs:
                toks.append(int(r.token))
                if r.finish_reason:
                    done += 1
        t_end = time.perf_counter()
        return {
            "ttft_s": (t_first - t_submit) if t_first is not None else None,
            "total_s": t_end - t_submit,
            "n_prompt": len(ids),
            "toks": toks,
        }

    for step_i, step in enumerate(plan):
        mode = step["mode"]
        label = step.get("label", "")
        n_repeat = step.get("n_repeat", 0)
        chunk = step.get("chunk", 2048)
        tag = step.get("tag", f"step{step_i}")
        sleep_before_s = step.get("sleep_before_s", 0)
        if sleep_before_s:
            if rank == 0:
                print(f"[e1] cooldown sleep {sleep_before_s}s before step {step_i+1}", flush=True)
            time.sleep(sleep_before_s)
        if rank == 0:
            print(f"[e1] === step {step_i+1}/{len(plan)} mode={mode} label={label} "
                  f"chunk={chunk} tag={tag} ===", flush=True)

        if mode == "timing":
            tail = "17 * 23 = (topic MVCC)"
            source = step.get("source", "repeat")
            file_recipe = None
            if source == "file":
                body_text, file_recipe = build_file_body(
                    step["file"], step.get("target_tokens", 13900), tok,
                    char_start=step.get("char_start", 0),
                    char_len=step.get("char_len"),
                )
                content = "참고 문서: " + body_text + " " + tail
                if rank == 0:
                    print(f"[e1] file-source recipe: {json.dumps(file_recipe, ensure_ascii=False)}",
                          flush=True)
            else:
                content = build_plain(n_repeat, tail)
            ids = tok.apply_chat_template([{"role": "user", "content": content}],
                                           add_generation_prompt=True)
            if rank == 0:
                print(f"[e1] timing label={label} source={source} n_repeat={n_repeat} "
                      f"n_prompt={len(ids)} chunk={chunk}", flush=True)
            if step.get("warmup"):
                warm_ids = ids[:256] if len(ids) > 256 else ids
                _ = run_once(warm_ids, 4, chunk)
                if rank == 0:
                    print("[e1] warmup done (미계측)", flush=True)
            reps = step.get("reps", 1)
            max_tokens = step.get("max_tokens", 8)
            for rep in range(reps):
                mx.reset_peak_memory()
                res = run_once(ids, max_tokens, chunk)
                peak_gb = round(mx.get_peak_memory() / 1e9, 3)
                if rank == 0 and res["ttft_s"]:
                    prefill_toks = res["n_prompt"]
                    toks_per_s = prefill_toks / res["ttft_s"]
                    out = {
                        "tag": tag, "mode": "timing", "label": label,
                        "world": world, "chunk": chunk, "rep": rep,
                        "source": source,
                        "n_prompt": prefill_toks, "ttft_s": round(res["ttft_s"], 4),
                        "prefill_tok_s": round(toks_per_s, 2),
                        "peak_mem_gb": peak_gb,
                        "out_tokens": res["toks"],
                    }
                    if file_recipe is not None:
                        out["file_recipe"] = file_recipe
                    print(f"[E1-RESULT] {json.dumps(out, ensure_ascii=False)}", flush=True)

        elif mode == "needle":
            code = step.get("secret_code", "8317")
            frac = step.get("needle_frac", 0.5)
            max_tokens = step.get("max_tokens", 16)
            content = build_needle(n_repeat, code, frac, QUESTION)
            ids = tok.apply_chat_template([{"role": "user", "content": content}],
                                           add_generation_prompt=True)
            if rank == 0:
                print(f"[e1] needle label={label} n_repeat={n_repeat} "
                      f"n_prompt={len(ids)} code={code} chunk={chunk}", flush=True)
            res = run_once(ids, max_tokens, chunk)
            if rank == 0:
                text = tok.decode(res["toks"]) if res["toks"] else ""
                recalled = code in text
                out = {
                    "tag": tag, "mode": "needle", "label": label,
                    "world": world, "chunk": chunk,
                    "n_prompt": res["n_prompt"], "code": code,
                    "ttft_s": round(res["ttft_s"], 4) if res["ttft_s"] else None,
                    "output_text": text, "recalled": recalled,
                }
                print(f"[E1-RESULT] {json.dumps(out, ensure_ascii=False)}", flush=True)
        else:
            raise ValueError(f"unknown mode {mode}")

    if rank == 0:
        print("[e1-pass]", flush=True)


if __name__ == "__main__":
    main()
