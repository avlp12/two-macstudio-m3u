import mlx.core as mx, time, sys
g = mx.distributed.init()
r = g.rank()
def bench(nel, iters=20):
    x = mx.ones((nel,), dtype=mx.float32); mx.eval(x)
    y = mx.distributed.all_sum(x, group=g); mx.eval(y)  # 워밍업
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = mx.distributed.all_sum(x, group=g); mx.eval(y)
    mx.synchronize()
    dt = (time.perf_counter() - t0) / iters
    nb = nel * 4
    if r == 0:
        algo = nb / dt / 1e9
        print(f"all_sum {nb>>20:4d} MB: {dt*1e3:8.2f} ms -> {algo:6.2f} GB/s algo", flush=True)
# 레이턴시
x = mx.ones((1024,), dtype=mx.float32); mx.eval(x)
for _ in range(5): mx.eval(mx.distributed.all_sum(x, group=g))
mx.synchronize()
t0 = time.perf_counter()
for _ in range(200): mx.eval(mx.distributed.all_sum(x, group=g))
mx.synchronize()
if r == 0: print(f"latency 4KB all_sum: {(time.perf_counter()-t0)/200*1e6:7.1f} us", flush=True)
for mb in (4, 32, 128, 512):
    bench(mb * 262144)
if r == 0: print("JBENCH-DONE", flush=True)
