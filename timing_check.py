#!/usr/bin/env python
import torch, time, os
from torch.utils import cpp_extension

device = "cuda"
torch.set_default_device(device)
batch_size, seq_len, emb_dim, num_heads = 16, 1024, 256, 8
head_dim = emb_dim // num_heads
q = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
k = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
v = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)

# time cpp_extension.load() overhead
t0 = time.perf_counter()
for _ in range(10):
    kernel = cpp_extension.load(
        name="flash_pytorch_extension",
        sources=["flash_v2_mma.cu"],
        extra_cflags=['-O3', '-g'],
        extra_include_paths=[],
        build_directory=os.path.join(os.getcwd(), 'build'),
        verbose=False,
    )
t1 = time.perf_counter()
print(f"cpp_extension.load x10 (cached): {(t1-t0)*1000:.1f} ms")

# time just the kernel launch (no python overhead in the loop)
kernel = cpp_extension.load(
    name="flash_pytorch_extension",
    sources=["flash_v2_mma.cu"],
    extra_cflags=['-O3', '-g'],
    extra_include_paths=[],
    build_directory=os.path.join(os.getcwd(), 'build'),
    verbose=False,
)
for _ in range(10):
    kernel.flash_attn_v2_mma(q, k, v)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(50):
    kernel.flash_attn_v2_mma(q, k, v)
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"kernel only x50 (incl. launch): {(t1-t0)*1000:.1f} ms  -> per call {(t1-t0)*1000/50:.3f} ms")

# single call wall including python + synchronize (like benchmark.py does)
torch.cuda.synchronize()
t0 = time.perf_counter()
o = kernel.flash_attn_v2_mma(q, k, v)
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"single call (perf_counter incl sync): {(t1-t0)*1000:.3f} ms")

# pure kernel gpu time via events
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
start.record()
o = kernel.flash_attn_v2_mma(q, k, v)
end.record()
torch.cuda.synchronize()
print(f"pure GPU time: {start.elapsed_time(end):.3f} ms")
