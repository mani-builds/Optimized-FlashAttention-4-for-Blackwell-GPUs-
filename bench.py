#!/usr/bin/env python

import math
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# This compiles on your 4080 automatically
minimal_attn = load(
    name='minimal_attn',
    sources=['main.cpp', 'flash.cu'],
    extra_cuda_cflags=['-O3', '-arch=sm_89']  # 4080 = Ada = sm_89
)

# Small test first (you can increase seq_len later)
batch_size = 4
n_head = 8
seq_len = 512          # try 1024, 2048 later
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd, device='cuda', dtype=torch.float32)
k = torch.randn(batch_size, n_head, seq_len, head_embd, device='cuda', dtype=torch.float32)
v = torch.randn(batch_size, n_head, seq_len, head_embd, device='cuda', dtype=torch.float32)

print("=== Running your tiled matmul + softmax kernel ===")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    out = minimal_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# Compare with PyTorch baseline
print("=== PyTorch SDPA baseline ===")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    out_ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print("Sanity check passed?", torch.allclose(out, out_ref, atol=1e-2))
