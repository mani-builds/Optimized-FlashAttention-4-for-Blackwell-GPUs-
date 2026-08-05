#!/usr/bin/env python
import os
import torch
import math
from torch.utils import cpp_extension
import os, ninja, pybind11

# Ensure you have a CUdA device available
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("FlashAttention requires a CUdA-capable GPU")

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


torch.set_default_device(device)
#
batch_size = 16 # B
seq_len = 1024  # No. of tokens # N
emb_dim = 256  # embedding shape # d_model
num_heads = 8 # No. of heads # H

head_dim = emb_dim // num_heads # d

q = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
k = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
v = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)

print("Running flash_v1_kernel...")
o = naive_attention(q,k,v)
print("Finished flash_v1_kernel.")

# Pytorch profiling
print("\nRunning PyTorch Profiler...")
with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes = True,
        with_stack=False,
) as prof:
    # Run Naive
    with torch.profiler.record_function('## NAIVE_ATTENTION ##'):
        naive_attention(q,k,v)

    # Run FlashAttention
    # with torch.profiler.record_function('## FLASH_ATTENTION ##'):
    #     pytorch_flash_attn2(q,k,v)

    # Run Custom CUDA flash v1
    # with torch.profiler.record_function('## Custom CUDA V1 ##'):
    #     flash_v1_kernel(q,k,v)

# Analyze Results
print("\n=== Profiler Results (Sorted by CUDA Time) ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

# Export chrome trace
prof.export_chrome_trace("naive_profile_v1.json")
print("\nTrace exported to 'naive_profile_v1.json'")
