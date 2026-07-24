#!/usr/bin/env python
import torch
from torch.utils import cpp_extension
import os, ninja, pybind11
import math
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import time

# Limit the precision (i.e 0.4771)
# torch.set_printoptions(precision=4)

# Ensure you have a CUdA device available
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("FlashAttention requires a CUdA-capable GPU")

torch.set_default_device(device)
#
batch_size = 16 # B
seq_len = 1024  # No. of tokens # N
emb_dim = 256  # embedding shape # d_model
num_heads = 8 # No. of heads # H

head_dim = emb_dim // num_heads # d

# z = self-attn(Q,K,V) = softmax(Q @ K.T / sqrt(d)) @ V
# Q, K, V: [B, H, N, d]
# z: [B, H, N, d]

def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
              ) -> torch.Tensor:
    B, H, N, d = K.shape

    scale = 1 / math.sqrt(d)

    # Q @ K.T
    # Reads Q and K from HBM and write result to S
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale # [B, H, N, N]

    # #mask
    # mask = torch.tril(torch.ones(N, N)).bool()
    # S = S.masked_fill(~mask, float('-inf'))

    # Softmax(S)
    # Read S from HBM and write P to HBM
    P = torch.softmax(S, dim=-1) # [B, H, N, N]

    # Read P and V from HBM and write O to HBM
    O = torch.matmul(P, V) # [B, H, N, N] @ [B, H, N, d] = [B, H, N, d]

    return O

def pytorch_flash_attn2(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                        ) -> torch.Tensor:

    # FlashAttn-2
    # Only enable flash attention backend
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        O = F.scaled_dot_product_attention(query=Q, key=K, value=V)

    return O

def flash_v1_kernel(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                        ) -> torch.Tensor:
    # CUdA Flash Kernel (flash_v1.cu)
    build_dir = os.path.join(os.getcwd(), 'build')
    os.makedirs(build_dir, exist_ok=True)
    kernel = cpp_extension.load(
        name = "flash_pytorch_extension",
        sources = ["flash_v1.cu"],
        extra_cflags=['-O3', '-g'],
        extra_include_paths=[pybind11.get_include()],
        build_directory=os.path.join(os.getcwd(), 'build'),
        verbose = True
    )

    O = kernel.flash_attn_v1(Q, K, V)
    return O


torch.manual_seed(123)

q = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
k = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
v = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)

o = naive_attention(q,k,v)
# print(o[:1,:1,:10, :10])
# torch.cuda.synchronize()
o1 = pytorch_flash_attn2(q,k,v)
# print(o1[:1,:1,:10, :10])
# torch.cuda.synchronize()
assert torch.allclose(o, o1, atol=1e-2), "Naive and Pytorch Outputs do not match"

o2 = flash_v1_kernel(q,k,v)
# print(o2[:1,:1,:10, :10])
# print(type(o2))
assert torch.allclose(o1, o2, atol=1e-2), "Pytorch and CUdA kernel outputs do not match"

# torch.cuda.synchronize() # Wait for check to finish completely

# Warmup (imp for hiding overhead latencies)
for _ in range(10):
    _ = naive_attention(q,k,v)
    _ = pytorch_flash_attn2(q,k,v)
    _ = flash_v1_kernel(q,k,v)
torch.cuda.synchronize()

# Simple Time Measurement
# start_time = time.perf_counter()
# naive_attention(q,k,v)
# torch.cuda.synchronize()
# end_time = time.perf_counter()
# naive_time = (end_time - start_time) * 1000

# start_time = time.perf_counter()
# pytorch_flash_attn2(q,k,v)
# torch.cuda.synchronize()
# end_time = time.perf_counter()
# flash_time = (end_time - start_time) * 1000

start_time = time.perf_counter()
flash_v1_kernel(q,k,v)
torch.cuda.synchronize()
end_time = time.perf_counter()
custom_v1_time = (end_time - start_time) * 1000

# print(f"Naive Attention Time: {naive_time:.3f} ms")
# print(f"Flash Attention Time: {flash_time:.3f} ms")
print(f"Custom Flash v1 Kernel Time: {custom_v1_time:.3f} ms")

# # Pytorch profiling
# print("\nRunning PyTorch Profiler...")
# with torch.profiler.profile(
#         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#         record_shapes = True,
#         with_stack=False,
# ) as prof:
#     # Run Naive
#     with torch.profiler.record_function('## NAIVE_ATTENTION ##'):
#         naive_attention(q,k,v)

#     # Run FlashAttention
#     with torch.profiler.record_function('## FLASH_ATTENTION ##'):
#         pytorch_flash_attn2(q,k,v)

#     # Run Custom CUDA flash v1
#     with torch.profiler.record_function('## Custom CUDA V1 ##'):
#         flash_v1_kernel(q,k,v)

# # Analyze Results
# print("\n=== Profiler Results (Sorted by CUdA Time) ===")
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

# # Export chrome trace
# prof.export_chrome_trace("flash_attn_profile_v2.2.json")
# print("\nTrace exported to 'flash_attn_profile_v2.2.json'")
