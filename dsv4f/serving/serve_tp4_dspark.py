#!/usr/bin/env python3
"""OpenAI-compatible TP4 DSpark server for DeepSeek-V4-Pro-0813.

Rank 0 owns HTTP. Ranks 1-N wait on a CPU TCP socket between requests and
only enter JACCL for a synchronized generate. Ring GEMM stays on the gather
path (PRO0813_DSPARK_RING=off). Prefix cache is best-effort: exact append
or a successful trim; otherwise the next turn prefills in full.

This is the workable singleton endpoint for DeepSeek Harness / Terminus.
It is not mlx_lm.server and not the omlx product server.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import signal
import socket
import struct
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True

MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_HTTP_BODY_BYTES = 16 * 1024 * 1024
THINK_START = "<think>"
THINK_END = "</think>"
DSML = "｜DSML｜"
BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
DEFAULT_SEED = 20260814
DEFAULT_MODEL = "/opt/models/DeepSeek-V4-Pro-0813-MXFP4-MLX"


class RequestError(ValueError):
    """Caller-facing HTTP 400."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def send_frame(sock: socket.socket, value: Any) -> None:
    payload = _json_bytes(value)
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError(f"control frame too large: {len(payload)}")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("control socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket) -> Any:
    length = struct.unpack("!I", _recv_exact(sock, 4))[0]
    if length > MAX_FRAME_BYTES:
        raise ValueError(f"control frame exceeds limit: {length}")
    return json.loads(_recv_exact(sock, length))


def configure_socket(sock: socket.socket) -> None:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    if hasattr(socket, "TCP_KEEPALIVE"):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 30)
    sock.settimeout(None)


def message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return str(content)


def normalize_messages(raw: list[Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RequestError("every message must be an object")
        role = item.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise RequestError(f"unsupported role {role!r}")
        msg = {"role": "system" if role == "developer" else role}
        msg["content"] = message_text(item.get("content"))
        if item.get("tool_calls"):
            msg["tool_calls"] = item["tool_calls"]
        if item.get("tool_call_id"):
            msg["tool_call_id"] = item["tool_call_id"]
        if item.get("name"):
            msg["name"] = item["name"]
        if item.get("reasoning_content"):
            msg["reasoning_content"] = item["reasoning_content"]
        messages.append(msg)
    if not messages:
        raise RequestError("messages must be a non-empty list")
    return messages


def validate_request(payload: Any, *, output_cap: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RequestError("request body must be a JSON object")
    stream = bool(payload.get("stream", False))
    if payload.get("n", 1) != 1:
        raise RequestError("only n=1 is supported")
    max_tokens = payload.get("max_completion_tokens", payload.get("max_tokens", 512))
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise RequestError("max_tokens must be a positive integer")
    if max_tokens > output_cap:
        raise RequestError(f"max_tokens={max_tokens} exceeds server cap {output_cap}")
    thinking_mode = payload.get("thinking_mode")
    if thinking_mode is None:
        enable = payload.get("enable_thinking", True)
        thinking_mode = "thinking" if enable else "chat"
    if thinking_mode not in ("chat", "thinking"):
        raise RequestError("thinking_mode must be 'chat' or 'thinking'")
    effort = payload.get("reasoning_effort")
    if effort is None and isinstance(payload.get("chat_template_kwargs"), dict):
        effort = payload["chat_template_kwargs"].get("reasoning_effort")
    if effort not in (None, "low", "high", "max"):
        raise RequestError("reasoning_effort must be low, high or max")
    temp = payload.get("temperature", 0.0)
    try:
        temp = float(temp)
    except (TypeError, ValueError) as exc:
        raise RequestError("temperature must be a number") from exc
    if temp < 0:
        raise RequestError("temperature must be >= 0")
    top_p = payload.get("top_p", 0.95)
    try:
        top_p = float(top_p)
    except (TypeError, ValueError) as exc:
        raise RequestError("top_p must be a number") from exc
    if not 0 < top_p <= 1:
        raise RequestError("top_p must be in (0, 1]")
    seed = DEFAULT_SEED
    if payload.get("seed") is not None:
        try:
            seed = int(payload["seed"])
        except (TypeError, ValueError) as exc:
            raise RequestError("seed must be an integer") from exc
        if seed < 0:
            raise RequestError("seed must be >= 0")
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise RequestError("messages must be a non-empty list")
    return {
        "messages": normalize_messages(messages),
        "max_tokens": max_tokens,
        "thinking_mode": thinking_mode,
        "reasoning_effort": effort or "low",
        "temperature": temp,
        "top_p": top_p,
        "tools": payload.get("tools"),
        "model": str(payload.get("model") or "deepseek-v4-pro-tp4"),
        "seed": seed,
        "stream": stream,
    }


def common_prefix_len(left: list[int], right: list[int]) -> int:
    n = min(len(left), len(right))
    i = 0
    while i < n and left[i] == right[i]:
        i += 1
    return i


def plan_cache(new_ids: list[int], cached_ids: Optional[list[int]]) -> dict[str, Any]:
    """Decide how much of the saved cache to keep. No MLX here."""
    if not cached_ids:
        return {"mode": "miss", "keep": 0, "trim": 0, "suffix": new_ids}
    keep = common_prefix_len(new_ids, cached_ids)
    if keep == 0:
        return {"mode": "miss", "keep": 0, "trim": 0, "suffix": new_ids}
    if keep == len(cached_ids):
        return {
            "mode": "append",
            "keep": keep,
            "trim": 0,
            "suffix": new_ids[keep:],
        }
    return {
        "mode": "trim",
        "keep": keep,
        "trim": len(cached_ids) - keep,
        "suffix": new_ids[keep:],
    }


_INVOKE = re.compile(
    rf"<{re.escape(DSML)}invoke name=\"([^\"]+)\">(.*?)</{re.escape(DSML)}invoke>",
    re.DOTALL,
)
_PARAM = re.compile(
    rf"<{re.escape(DSML)}parameter name=\"([^\"]+)\"(?:\s+string=\"(true|false)\")?>(.*?)</{re.escape(DSML)}parameter>",
    re.DOTALL,
)


def parse_dsml_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for index, match in enumerate(_INVOKE.finditer(text)):
        name = match.group(1)
        body = match.group(2)
        params: dict[str, Any] = {}
        for param in _PARAM.finditer(body):
            raw = param.group(3).strip()
            if param.group(2) == "false":
                try:
                    params[param.group(1)] = json.loads(raw)
                except json.JSONDecodeError:
                    params[param.group(1)] = raw
            else:
                params[param.group(1)] = raw
        if not params and body.strip():
            arguments = body.strip()
        else:
            arguments = json.dumps(params, ensure_ascii=False)
        calls.append(
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return calls


def parse_assistant(text: str, *, thinking_mode: str, hit_eos: bool) -> dict[str, Any]:
    reasoning = ""
    content = (
        text.replace(EOS_TOKEN, "").replace(BOS_TOKEN, "").strip()
    )
    if THINK_END in text:
        reasoning, _, content = text.partition(THINK_END)
        reasoning = reasoning.replace(THINK_START, "").strip()
        content = content.lstrip("\n")
    elif text.startswith(THINK_START) and thinking_mode == "thinking":
        reasoning = text[len(THINK_START) :].strip()
        content = ""
    tool_calls = parse_dsml_tool_calls(content)
    if tool_calls:
        start = content.find(f"<{DSML}tool_calls>")
        if start < 0:
            start = content.find(f"<{DSML}invoke")
        if start >= 0:
            content = content[:start].rstrip()
    finish = "length"
    if hit_eos:
        finish = "tool_calls" if tool_calls else "stop"
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": tool_calls,
        "finish_reason": finish,
    }


def encode_prompt(tokenizer, request: dict[str, Any]) -> list[int]:
    kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "thinking_mode": request["thinking_mode"],
        "reasoning_effort": request["reasoning_effort"],
    }
    if request.get("tools"):
        kwargs["tools"] = request["tools"]
    encoded = tokenizer.apply_chat_template(request["messages"], **kwargs)
    if isinstance(encoded, str):
        return list(tokenizer.encode(encoded, add_special_tokens=False))
    return list(encoded)


class WorkerControl:
    def __init__(self, world: int, bind_host: str, port: int):
        self.world = world
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((bind_host, port))
        self.listener.listen(world - 1)
        self.listener.settimeout(180)
        self.workers: dict[int, socket.socket] = {}

    def accept_all(self) -> None:
        while len(self.workers) < self.world - 1:
            conn, _ = self.listener.accept()
            configure_socket(conn)
            hello = recv_frame(conn)
            rank = hello.get("rank") if isinstance(hello, dict) else None
            if (
                not isinstance(hello, dict)
                or hello.get("kind") != "hello"
                or rank not in range(1, self.world)
            ):
                conn.close()
                raise RuntimeError(f"invalid worker handshake: {hello!r}")
            if rank in self.workers:
                conn.close()
                raise RuntimeError(f"duplicate worker rank {rank}")
            self.workers[rank] = conn
            send_frame(conn, {"kind": "hello-ack", "rank": rank})

    def dispatch(self, command: dict[str, Any]) -> None:
        for rank in sorted(self.workers):
            send_frame(self.workers[rank], command)

    def collect_kind(self, kind: str) -> dict[int, Any]:
        replies: dict[int, Any] = {}
        for rank in sorted(self.workers):
            reply = recv_frame(self.workers[rank])
            if reply.get("kind") != kind or reply.get("rank") != rank:
                raise RuntimeError(f"expected {kind} from rank {rank}, got {reply!r}")
            replies[rank] = reply
        return replies

    def collect(self, expected_signature: list[int]) -> None:
        replies = self.collect_kind("done")
        for rank, reply in replies.items():
            if reply.get("signature") != expected_signature:
                raise RuntimeError(
                    f"rank {rank} signature differs: {reply.get('signature')} "
                    f"!= {expected_signature}"
                )

    def close(self) -> None:
        for conn in self.workers.values():
            try:
                send_frame(conn, {"kind": "stop"})
            except Exception:
                pass
            conn.close()
        self.listener.close()


def connect_worker(host: str, port: int, rank: int) -> socket.socket:
    deadline = time.monotonic() + 180
    last_error = None
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(5)
            sock.connect((host, port))
            configure_socket(sock)
            send_frame(sock, {"kind": "hello", "rank": rank})
            reply = recv_frame(sock)
            if reply != {"kind": "hello-ack", "rank": rank}:
                raise RuntimeError(f"invalid coordinator handshake: {reply!r}")
            return sock
        except Exception as error:
            last_error = error
            sock.close()
            time.sleep(1)
    raise RuntimeError(f"rank {rank} could not connect to control host: {last_error}")


@dataclass
class WorkItem:
    payload: dict[str, Any]
    reply: queue.Queue


class RankZeroRuntime:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.requests: queue.Queue[WorkItem] = queue.Queue(maxsize=1)
        self.busy = threading.Lock()
        self.ready = threading.Event()

    def submit(self, payload: dict[str, Any], timeout: float = 3600) -> dict[str, Any]:
        if not self.ready.is_set():
            raise RuntimeError("model is not ready")
        if not self.busy.acquire(blocking=False):
            raise RequestError("server already has an active request")
        try:
            replies: queue.Queue = queue.Queue(maxsize=1)
            self.requests.put(WorkItem(payload=payload, reply=replies), timeout=1)
            outcome = replies.get(timeout=timeout)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        finally:
            self.busy.release()


def build_handler(runtime: RankZeroRuntime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DeepSeekV4ProDSpark/1"

        def log_message(self, fmt, *args):
            print(f"[pro0813-http] {self.address_string()} {fmt % args}", flush=True)

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_sse(self, completion: dict[str, Any]) -> None:
            """Replay a finished completion as OpenAI/DeepSeek SSE.

            DSpark generate is still one blocking call. The harness always
            sends stream=true and requires a [DONE]-terminated event stream,
            so we emit the assembled message as deltas after the fact.
            """
            choice = completion["choices"][0]
            message = choice["message"]
            finish = choice.get("finish_reason") or "stop"
            created = completion.get("created", int(time.time()))
            model = completion.get("model", "deepseek-v4-pro-tp4")
            req_id = completion.get("id", "chatcmpl-local")

            def event(delta: dict[str, Any], finish_reason: Any = None, usage: Any = None) -> bytes:
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                if usage is not None:
                    chunk["usage"] = usage
                return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.write(event({"role": "assistant"}))
            reasoning = message.get("reasoning_content") or ""
            if reasoning:
                self.wfile.write(event({"reasoning_content": reasoning}))
            content = message.get("content") or ""
            if content:
                self.wfile.write(event({"content": content}))
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                streamed = []
                for index, call in enumerate(tool_calls):
                    item = dict(call)
                    item["index"] = index
                    streamed.append(item)
                self.wfile.write(event({"tool_calls": streamed}))
            self.wfile.write(event({}, finish, completion.get("usage")))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                ready = runtime.ready.is_set()
                self.send_json(
                    HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                    {"status": "ok" if ready else "loading"},
                )
            elif self.path == "/v1/models":
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": runtime.model_name,
                                "object": "model",
                                "owned_by": "local",
                            }
                        ],
                    },
                )
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
                return
            try:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except (TypeError, ValueError) as exc:
                    raise RequestError("invalid Content-Length") from exc
                if length < 1 or length > MAX_HTTP_BODY_BYTES:
                    raise RequestError("invalid request body size")
                try:
                    payload = json.loads(self.rfile.read(length))
                except json.JSONDecodeError as exc:
                    raise RequestError("request body is not valid JSON") from exc
                result = runtime.submit(payload)
                if payload.get("stream"):
                    self.send_sse(result)
                else:
                    self.send_json(HTTPStatus.OK, result)
            except RequestError as error:
                self.send_json(
                    HTTPStatus.BAD_REQUEST, {"error": {"message": str(error)}}
                )
            except queue.Empty:
                self.send_json(
                    HTTPStatus.GATEWAY_TIMEOUT,
                    {"error": {"message": "model timed out"}},
                )
            except Exception as error:
                traceback.print_exc()
                self.send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": {"message": f"model request failed: {error}"}},
                )

    return Handler


@dataclass
class CacheSlot:
    token_ids: list[int] = field(default_factory=list)
    cache: Any = None


def realize_cache(cache: Any) -> None:
    if not cache:
        return
    import mlx.core as mx

    blobs = []
    for item in cache:
        state = getattr(item, "state", None)
        if state is not None:
            blobs.append(state)
    if blobs:
        mx.eval(*blobs)


_MODE_CODE = {"miss": 0, "append": 1, "trim": 2}


def insert_pieces(prompt_ids: list[int], plan: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Return (suffix, cached_prefix) for BatchGenerator.insert."""
    if plan["mode"] == "miss" or not plan["keep"]:
        return prompt_ids, []
    keep = int(plan["keep"])
    suffix = list(plan["suffix"])
    if suffix:
        return suffix, prompt_ids[:keep]
    if keep < 1:
        return prompt_ids, []
    return [prompt_ids[keep - 1]], prompt_ids[: keep - 1]


def try_apply_plan(slot: CacheSlot, plan: dict[str, Any]) -> tuple[Any, str]:
    if plan["mode"] == "miss" or slot.cache is None:
        return None, "miss"
    if plan["mode"] == "append":
        return slot.cache, "append"
    try:
        from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
    except Exception:
        return None, "miss"
    if not can_trim_prompt_cache(slot.cache):
        return None, "miss"
    trimmed = trim_prompt_cache(slot.cache, int(plan["trim"]))
    if not trimmed:
        return None, "miss"
    return slot.cache, "trim"


def agree_plan(local: dict[str, Any], group) -> dict[str, Any]:
    """All-gather the cache plan. Any disagreement becomes a coordinated miss."""
    import mlx.core as mx

    packed = mx.array(
        [
            _MODE_CODE[local["mode"]],
            int(local["keep"]),
            int(len(local["suffix"])),
        ],
        dtype=mx.uint32,
    )
    gathered = mx.distributed.all_gather(packed, group=group).reshape(group.size(), -1)
    mx.eval(gathered)
    if bool(mx.all(gathered == gathered[0]).item()):
        return local
    return {"mode": "miss", "keep": 0, "trim": 0, "suffix": None}


def agree_ok(ok: bool, group) -> bool:
    import mlx.core as mx

    flag = mx.array([1 if ok else 0], dtype=mx.uint32)
    gathered = mx.distributed.all_gather(flag, group=group).reshape(group.size(), -1)
    mx.eval(gathered)
    return bool(mx.all(gathered == 1).item())


def tokenizer_eos_ids(tokenizer) -> list[int]:
    ids: list[int] = []
    raw = getattr(tokenizer, "eos_token_ids", None)
    if raw:
        ids.extend(int(x) for x in raw)
    single = getattr(tokenizer, "eos_token_id", None)
    if single is not None:
        ids.append(int(single))
    # Unique, stable order.
    seen: set[int] = set()
    out: list[int] = []
    for item in ids:
        if item not in seen and item >= 0:
            seen.add(item)
            out.append(item)
    return out


def generate_dspark(
    model,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    prefill_step: int,
    slot: CacheSlot,
    group,
    eos_ids: list[int],
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    from dspark_tp4_common import MtpLogTap, snapshot_mtp

    if not prompt_ids:
        raise RequestError("encoded prompt is empty")

    plan = agree_plan(plan_cache(prompt_ids, slot.token_ids), group)
    apply_ok = True
    reuse = None
    cache_mode = "miss"
    if plan["mode"] != "miss":
        try:
            reuse, cache_mode = try_apply_plan(slot, plan)
            if reuse is None:
                apply_ok = False
        except Exception:
            apply_ok = False
            reuse = None
    if not agree_ok(apply_ok and (plan["mode"] == "miss" or reuse is not None), group):
        slot.cache = None
        slot.token_ids = []
        plan = {"mode": "miss", "keep": 0, "trim": 0, "suffix": prompt_ids}
        reuse = None
        cache_mode = "miss"

    suffix, cached_prefix = insert_pieces(prompt_ids, plan if reuse is not None else {"mode": "miss", "keep": 0, "suffix": prompt_ids})
    caches = [reuse] if reuse is not None else None
    all_tokens = [cached_prefix] if reuse is not None else None

    sampler = (
        make_sampler(temp=0.0)
        if temperature <= 0
        else make_sampler(temp=temperature, top_p=top_p)
    )
    tap = MtpLogTap()
    logger = logging.getLogger("omlx.patches.mlx_lm_mtp.batch_generator")
    logger.addHandler(tap)
    stop_tokens = [[int(eid)] for eid in eos_ids] if eos_ids else None
    batch_gen = BatchGenerator(
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        stop_tokens=stop_tokens,
        completion_batch_size=1,
        prefill_batch_size=1,
        prefill_step_size=prefill_step,
    )
    tokens: list[int] = []
    stats: dict[str, Any] = {}
    finish = None
    saved_cache = None
    saved_all: list[int] = []
    t0 = time.monotonic()
    first = None
    last = None
    try:
        mx.random.seed(int(seed))
        insert_kwargs: dict[str, Any] = {
            "max_tokens": [max_tokens],
        }
        if caches is not None:
            insert_kwargs["caches"] = caches
            insert_kwargs["all_tokens"] = all_tokens
        batch_gen.insert([suffix], **insert_kwargs)
        while True:
            responses = batch_gen.next_generated()
            if not responses:
                break
            now = time.monotonic()
            for response in responses:
                if first is None:
                    first = now
                last = now
                tokens.append(int(response.token))
                live = snapshot_mtp(batch_gen)
                if live:
                    stats = live
                if response.finish_reason:
                    finish = response.finish_reason
                    if response.prompt_cache is not None:
                        saved_cache = response.prompt_cache
                        saved_all = list(response.all_tokens or [])
                    break
            if finish:
                break
        live = snapshot_mtp(batch_gen)
        if live:
            stats = live
    finally:
        try:
            batch_gen.close()
        except Exception:
            pass
        logger.removeHandler(tap)

    if saved_cache is None and tokens:
        # Fallback: keep the last known cache even if the finish payload
        # omitted it (should not happen on a normal length/stop finish).
        saved_all = (slot.token_ids[: plan["keep"]] if reuse is not None else []) + suffix + tokens
    else:
        saved_all = saved_all or (
            (prompt_ids if reuse is None else prompt_ids[: plan["keep"]] + suffix) + tokens
        )

    if eos_ids and tokens and tokens[-1] in eos_ids:
        tokens = tokens[:-1]
        finish = finish or "stop"

    slot.cache = saved_cache
    slot.token_ids = list(saved_all)
    realize_cache(slot.cache)

    hit_eos = finish == "stop"
    ttft = (first - t0) if first else None
    decode_s = (last - first) if first and last else None
    decode_tokens = max(len(tokens) - 1, 0)
    return {
        "token_ids": tokens,
        "hit_eos": hit_eos,
        "finish": finish or "length",
        "ttft_s": ttft,
        "decode_s": decode_s,
        "total_s": time.monotonic() - t0,
        "prompt_tokens": len(prompt_ids),
        "cached_tokens": plan["keep"] if cache_mode != "miss" else 0,
        "cache_mode": cache_mode,
        "prefill_tokens": len(suffix) if cache_mode != "miss" else len(prompt_ids),
        "decode_tps": (decode_tokens / decode_s) if decode_s and decode_tokens else None,
        "mtp": stats,
        "mtp_activated": sum(
            1 for line in tap.lines if line.startswith("MTP path activated")
        ),
    }


def output_signature(token_ids: list[int]):
    import mlx.core as mx

    digest = 0
    for token in token_ids:
        digest = (digest * 131 + int(token)) & 0xFFFFFFFF
    return mx.array([len(token_ids), digest], dtype=mx.uint32)


def worker_loop(sock, rank, model, group, slot: CacheSlot, prefill_step: int) -> None:
    import mlx.core as mx

    from dspark_tp4_common import consensus

    while True:
        command = recv_frame(sock)
        if command.get("kind") == "stop":
            return
        if command.get("kind") != "generate":
            raise RuntimeError(f"rank {rank} invalid command: {command!r}")
        send_frame(sock, {"kind": "armed", "rank": rank})
        consensus(mx.array([1], dtype=mx.uint32), group, "generate-arm")
        error = ""
        try:
            result = generate_dspark(
                model,
                command["prompt_ids"],
                max_tokens=command["max_tokens"],
                temperature=command["temperature"],
                top_p=command["top_p"],
                seed=command["seed"],
                prefill_step=prefill_step,
                slot=slot,
                group=group,
                eos_ids=list(command.get("eos_ids") or []),
            )
        except Exception as exc:
            slot.cache = None
            slot.token_ids = []
            result = {"token_ids": []}
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        signature = output_signature(result["token_ids"])
        ok = mx.array([0 if error else 1], dtype=mx.uint32)
        gathered = mx.distributed.all_gather(ok, group=group).reshape(group.size(), -1)
        mx.eval(gathered)
        if not bool(mx.all(gathered == 1).item()):
            slot.cache = None
            slot.token_ids = []
            send_frame(
                sock,
                {
                    "kind": "done",
                    "rank": rank,
                    "signature": [0, 0],
                    "error": error or "peer failed",
                },
            )
            continue
        consensus(signature, group, f"request {command['request_id']}")
        send_frame(
            sock,
            {
                "kind": "done",
                "rank": rank,
                "signature": signature.tolist(),
            },
        )


def rank_zero_loop(runtime, control, model, tokenizer, group, args, slot: CacheSlot) -> None:
    import mlx.core as mx

    from dspark_tp4_common import consensus

    runtime.ready.set()
    print(
        f"[pro0813-server-ready] model={args.model_name} "
        f"http={args.serve_host}:{args.serve_port} "
        f"control_workers={len(control.workers)} ring=off",
        flush=True,
    )
    while True:
        item = runtime.requests.get()
        try:
            request = validate_request(item.payload, output_cap=args.max_output_tokens)
            prompt_ids = encode_prompt(tokenizer, request)
            if len(prompt_ids) > args.max_context_tokens:
                raise RequestError(
                    f"prompt has {len(prompt_ids)} tokens; cap is {args.max_context_tokens}"
                )
            request_id = uuid.uuid4().hex
            eos_ids = tokenizer_eos_ids(tokenizer)
            command = {
                "kind": "generate",
                "request_id": request_id,
                "prompt_ids": prompt_ids,
                "max_tokens": request["max_tokens"],
                "temperature": request["temperature"],
                "top_p": request["top_p"],
                "seed": request["seed"],
                "eos_ids": eos_ids,
            }
            control.dispatch(command)
            control.collect_kind("armed")
            consensus(mx.array([1], dtype=mx.uint32), group, "generate-arm")
            error = ""
            try:
                result = generate_dspark(
                    model,
                    prompt_ids,
                    max_tokens=request["max_tokens"],
                    temperature=request["temperature"],
                    top_p=request["top_p"],
                    seed=request["seed"],
                    prefill_step=args.prefill_step,
                    slot=slot,
                    group=group,
                    eos_ids=eos_ids,
                )
            except Exception as exc:
                slot.cache = None
                slot.token_ids = []
                result = {"token_ids": []}
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            signature = output_signature(result["token_ids"])
            ok = mx.array([0 if error else 1], dtype=mx.uint32)
            gathered = mx.distributed.all_gather(ok, group=group).reshape(
                group.size(), -1
            )
            mx.eval(gathered)
            if not bool(mx.all(gathered == 1).item()):
                slot.cache = None
                slot.token_ids = []
                control.collect_kind("done")
                raise RuntimeError(error or "generate failed on a peer rank")
            consensus(signature, group, f"request {request_id}")
            control.collect(signature.tolist())
            text = tokenizer.decode(result["token_ids"])
            message = parse_assistant(
                text,
                thinking_mode=request["thinking_mode"],
                hit_eos=result["hit_eos"],
            )
            finish_reason = message.pop("finish_reason")
            body = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request["model"],
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish_reason}
                ],
                "usage": {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": len(result["token_ids"]),
                    "total_tokens": result["prompt_tokens"] + len(result["token_ids"]),
                },
                "timings": {
                    "ttft_s": result["ttft_s"],
                    "decode_s": result["decode_s"],
                    "total_s": result["total_s"],
                    "decode_tps": result["decode_tps"],
                    "cache_mode": result["cache_mode"],
                    "cached_tokens": result["cached_tokens"],
                    "prefill_tokens": result["prefill_tokens"],
                    "mtp_activated": result["mtp_activated"],
                    "accept_pct": (result.get("mtp") or {}).get("accept_pct"),
                },
            }
            print(
                f"[pro0813-request] id={request_id} prompt={result['prompt_tokens']} "
                f"completion={len(result['token_ids'])} cache={result['cache_mode']}/"
                f"{result['cached_tokens']} ttft={result['ttft_s']} "
                f"decode_tps={result['decode_tps']} mtp={result['mtp_activated']} "
                f"accept={(result.get('mtp') or {}).get('accept_pct')}",
                flush=True,
            )
            item.reply.put(body)
        except Exception as error:
            item.reply.put(error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-name", default="deepseek-v4-pro-tp4")
    parser.add_argument("--serve-host", default="0.0.0.0")
    parser.add_argument("--serve-port", type=int, default=8095)
    parser.add_argument("--control-host", required=True)
    parser.add_argument("--control-bind", default="0.0.0.0")
    parser.add_argument("--control-port", type=int, default=18095)
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--prefill-step", type=int, default=512)
    parser.add_argument("--require-world", type=int, default=4)
    parser.add_argument("--depth", type=int, default=5)
    args = parser.parse_args()
    if args.max_context_tokens < 1 or args.max_output_tokens < 1:
        parser.error("token caps must be positive")
    if args.prefill_step < 1:
        parser.error("prefill-step must be positive")
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MLX_METAL_FAST_SYNCH", "0")
    os.environ["PRO0813_DSPARK_RING"] = "off"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import mlx.core as mx

    from dspark_tp4_common import load_sharded_dspark

    group = mx.distributed.init()
    rank = group.rank()
    world = group.size()
    if world != args.require_world:
        raise RuntimeError(f"server requires world={args.require_world}, got {world}")

    if rank == 0:
        print(
            f"[pro0813-server] loading {args.model} world={world} depth={args.depth}",
            flush=True,
        )
    model, tokenizer, load_s = load_sharded_dspark(args.model, group, depth=args.depth)
    slot = CacheSlot()
    if rank == 0:
        print(f"[pro0813-server] ready in {load_s:.1f}s", flush=True)

    if rank:
        sock = connect_worker(args.control_host, args.control_port, rank)
        try:
            worker_loop(sock, rank, model, group, slot, args.prefill_step)
        finally:
            sock.close()
        return

    control = WorkerControl(world, args.control_bind, args.control_port)
    runtime = RankZeroRuntime(args.model_name)
    http_server = ThreadingHTTPServer(
        (args.serve_host, args.serve_port), build_handler(runtime)
    )
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_started = False

    def stop_handler(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    try:
        control.accept_all()
        http_thread.start()
        http_started = True
        rank_zero_loop(runtime, control, model, tokenizer, group, args, slot)
    except KeyboardInterrupt:
        print("[pro0813-server-stop] signal received", flush=True)
    finally:
        runtime.ready.clear()
        if http_started:
            http_server.shutdown()
        http_server.server_close()
        control.close()


if __name__ == "__main__":
    main()
