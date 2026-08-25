"""B=1 비-배치 캐시 우회 — BatchGenerator 프리필의 '배치-캐시 세(1.53×)' 제거.

문제([I-btax1])
  mlx-lm `BatchGenerator` 는 `PromptProcessingBatch.__init__` → `_merge_caches()`
  에서 batch_size 와 무관하게 전 캐시를 Batch* 변종으로 바꾼다
  (RotatingKVCache→BatchRotatingKVCache, PoolingCache→BatchPoolingCache).
  Batch* 변종의 `offset` 은 python int 가 아니라 **mx.array** 라서, DSv4 의
  융합 프리필 커널 3종이 전부 첫 줄에서 fallback 한다:

    wsdpa_attention.wsdpa_prefill:288       `if isinstance(offset, mx.array): return None`
    wsdpa_attention.wsdpa_topk_prefill:351  같은 가드
    deepseek_v4_model._sparse_pooled_attention:711
                                            `and not isinstance(q_offset, mx.array)`

  → 43층 중 41층(compress_ratio 4 × 21, 128 × 20)이 스톡 SDPA/게더 경로로 떨어진다.
  1box 13.9K·chunk2048 실측: raw 청크루프 605 tok/s vs BatchGenerator 395 tok/s.

해소
  prefill/decode 배치가 **단일 시퀀스**일 때는 Batch* 로 바꿀 이유가 없다.
  `_merge_caches([one])` 를 우회해 plain 캐시를 그대로 쓰면 offset 이 int 로
  유지되어 raw 청크루프와 **동일한 코드 경로**를 탄다(수치 동일이 구조적 보장).
  Batch* 전용 API 4종은 B=1 의미의 shim 으로 채우고, 두 번째 시퀀스가 합류하는
  순간 `_extend_cache` 에서 `merge()` 로 Batch* 승격한다(연속배칭 무손상).

실측(1box, box A, 2026-08-25 — logs/btax_13k9.log)
  A  raw 청크루프(융합 ON)   605.5-606.8 tok/s
  B  BatchGenerator 스톡      → 본 패치 없이
  C  BatchGenerator + 본 패치 → A 근접, greedy 토큰 100% 일치

두 변종
  P1  (기본)                       프리필+디코드 모두 plain 유지.
  P1b DSV4_B1_PROMOTE_AT_HANDOFF=1 프리필→디코드 전환에서 Batch* 로 되돌린다.
      디코드 경로(특히 MTP verify 의 L>1 블록)가 스톡과 완전 동일해져
      [I296] 계열 위험을 프리필로 한정한다. 비용 = 전환 시 merge 1회.

정합성 주의
  본 패치는 **비트 동일이 아니라 fp-동치**다. 41개 층의 어텐션이 스톡 SDPA →
  융합커널로 바뀌므로 로짓이 fp 수준에서 달라지고, greedy 연속 생성은 프롬프트에
  따라 수 토큰 뒤 갈라질 수 있다(실측: 13.9K/8토큰·256tok/48토큰·602tok/24토큰은
  전부 동일, 1323tok/24토큰은 index 7에서 분기). 바뀌는 방향은 raw 청크루프
  (=E2 ref1box 기준선)와 **같아지는** 쪽이다.

킬스위치
  DSV4_B1_CACHE_BYPASS=0 이면 apply() 가 아무 것도 하지 않는다.

사용
  apply_deepseek_v4_patch() (+ apply_mlx_lm_mtp_patch()) **뒤**, BatchGenerator
  생성 **전**에 한 번:
      import b1_cache_bypass; b1_cache_bypass.apply()
"""

from __future__ import annotations

import copy
import importlib
import logging
import os

logger = logging.getLogger(__name__)

_APPLIED = False
_PLAIN_NAMES = ("RotatingKVCache", "PoolingCache", "KVCache")


def _install_plain_cache_shims() -> None:
    """plain 캐시에 Batch* 전용 API 를 심는다 (의미: 단일 시퀀스 전용).

    mlx-lm/omlx 어디에도 이 이름들에 대한 hasattr 분기가 없으므로(2026-08-25 확인)
    클래스에 메서드를 추가해도 다른 경로의 동작은 변하지 않는다.
    """
    from mlx_lm.models.cache import PoolingCache, RotatingKVCache

    def _filter(self, batch_indices):
        idx = list(batch_indices)
        if idx != [0]:
            raise RuntimeError(f"plain cache filter({idx}): B=1 전용")

    def _extract(self, idx):
        if idx != 0:
            raise RuntimeError(f"plain cache extract({idx}): B=1 전용")
        # 스냅숏은 이후 in-place 링버퍼 쓰기와 분리돼야 한다 → 깊은 복사.
        # (스톡 BatchRotatingKVCache.extract 도 새 RotatingKVCache 를 만들어 준다)
        return copy.deepcopy(self)

    def _prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None and max(left_padding) > 0:
            raise RuntimeError("plain cache: left padding 불가")
        if right_padding is not None and max(right_padding) > 0:
            raise RuntimeError("plain cache: right padding 불가")

    def _finalize(self):
        pass

    def _extend(self, other):
        # 정상 경로에서는 아래 extend_cache 가 먼저 Batch* 로 승격하므로 도달 불가.
        raise RuntimeError("plain cache extend: _extend_cache 승격 경로 누락")

    for cls in (RotatingKVCache, PoolingCache):
        if getattr(cls, "_b1_bypass_shim", False):
            continue
        cls.filter = _filter
        cls.extract = _extract
        cls.prepare = _prepare
        cls.finalize = _finalize
        cls.extend = _extend
        cls._b1_bypass_shim = True


def _is_plain(c) -> bool:
    from mlx_lm.models.cache import CacheList

    if isinstance(c, CacheList):
        return any(_is_plain(x) for x in c.caches)
    return type(c).__name__ in _PLAIN_NAMES


def apply() -> bool:
    """`_merge_caches` B=1 우회 + `_extend_cache` 지연 승격. 멱등."""
    global _APPLIED
    if _APPLIED:
        return False
    if os.environ.get("DSV4_B1_CACHE_BYPASS", "1") != "1":
        logger.info("b1_cache_bypass disabled by DSV4_B1_CACHE_BYPASS")
        return False

    # mlx_lm.__init__ 이 generate 를 함수로 재노출해 속성 접근을 가리므로
    # 반드시 import_module 로 실제 모듈 객체를 잡는다.
    gen = importlib.import_module("mlx_lm.generate")

    _install_plain_cache_shims()
    orig_merge = gen._merge_caches
    orig_extend = gen._extend_cache

    def merge_caches(caches):
        if len(caches) == 1:
            return list(caches[0])          # plain 유지 → int offset → 융합커널 ON
        return orig_merge(caches)

    def extend_cache(cache_a, cache_b):
        if not cache_a:
            return cache_b
        if not cache_b:
            return cache_a
        if any(_is_plain(c) for c in cache_a) or any(_is_plain(c) for c in cache_b):
            # 두 번째 시퀀스 합류 → 여기서 Batch* 승격 (plain 은 B=1 전용)
            a = [type(c).merge([c]) if _is_plain(c) else c for c in cache_a]
            b = [type(c).merge([c]) if _is_plain(c) else c for c in cache_b]
            for ca, cb in zip(a, b):
                ca.extend(cb)
            return a
        return orig_extend(cache_a, cache_b)

    promote = os.environ.get("DSV4_B1_PROMOTE_AT_HANDOFF", "0") == "1"
    if promote:
        # P1b — 프리필→디코드 전환에서 Batch* 로 되돌려 디코드 경로를 스톡과 동일하게.
        ppb = gen.PromptProcessingBatch
        orig_generate = ppb.generate

        def generate(self, tokens):
            # 스톡 generate() 의 첫 두 줄을 선행 수행한 뒤 승격 →
            # GenerationBatch 는 처음부터 Batch* 캐시만 본다.
            if any(len(t) > 1 for t in tokens):
                self.prompt([t[:-1] for t in tokens])
                tokens = [t[-1:] for t in tokens]
            if any(_is_plain(c) for c in self.prompt_cache):
                self.prompt_cache = [
                    type(c).merge([c]) if _is_plain(c) else c
                    for c in self.prompt_cache
                ]
            return orig_generate(self, tokens)

        ppb.generate = generate

    gen._merge_caches = merge_caches
    gen._extend_cache = extend_cache
    _APPLIED = True
    logger.info("b1_cache_bypass applied (promote_at_handoff=%s)", promote)
    return True
