"""TB5 link bandwidth bench for the MLX ring backend.

Run via mlx.launch with MLX_HOSTFILE set by the launcher. Measures all_sum
algorithmic bandwidth (payload bytes / wall time) across message sizes, plus
small-message latency. Rank 0 prints a summary line per size.
"""

import time

import mlx.core as mx

WARMUP = 5
SIZES_MB = [4, 32, 128, 512]
LAT_BYTES = 4096
ITERS_FOR = {4: 40, 32: 20, 128: 10, 512: 6}


def sync_barrier(group):
    mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group))


def bench_size(group, nbytes, iters):
    x = mx.ones((nbytes // 4,), dtype=mx.float32)
    mx.eval(x)
    for _ in range(WARMUP):
        mx.eval(mx.distributed.all_sum(x, group=group))
    sync_barrier(group)
    tic = time.perf_counter()
    for _ in range(iters):
        x = mx.distributed.all_sum(x, group=group)
        mx.eval(x)
    toc = time.perf_counter()
    dt = (toc - tic) / iters
    return dt


def main():
    group = mx.distributed.init(backend="ring")
    rank, size = group.rank(), group.size()
    assert size == 2, f"expected 2 ranks, got {size}"
    sync_barrier(group)

    # latency: tiny all_sum round
    lat = bench_size(group, LAT_BYTES, 100)
    if rank == 0:
        print(f"latency  {LAT_BYTES}B all_sum: {lat*1e6:9.1f} us")

    for mb in SIZES_MB:
        nbytes = mb * 1024 * 1024
        dt = bench_size(group, nbytes, ITERS_FOR[mb])
        gbps = nbytes / dt / 1e9
        if rank == 0:
            print(f"all_sum {mb:4d} MB: {dt*1e3:8.2f} ms  ->  {gbps:6.2f} GB/s algo ({gbps*8:6.1f} Gbit/s)")

    sync_barrier(group)


if __name__ == "__main__":
    main()
