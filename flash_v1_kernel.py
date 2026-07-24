#!/usr/bin/env python
import os
import torch
from torch.utils import cpp_extension
import os, ninja, pybind11

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

q = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
k = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
v = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)

print("Running flash_v1_kernel...")
o = flash_v1_kernel(q,k,v)
print("Finished flash_v1_kernel.")
