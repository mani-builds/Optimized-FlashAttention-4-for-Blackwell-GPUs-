#!/usr/bin/env python
import torch
import math
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Ensure you have a CUDA device available
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("FlashAttention requires a CUDA-capable GPU")

torch.set_default_device(device)

batch_size = 16 # B
seq_len = 1024  # No. of tokens # N
emb_dim = 512 # embedding shape # d_model
num_heads = 8 # No. of heads # D

head_dim = emb_dim // num_heads # H

# z = self-attn(Q,K,V) = softmax(Q @ K.T / sqrt(d)) @ V
# Q, K, V: [B, H, N, D]
# z: [B, H, N, D]

def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
              ) -> torch.Tensor:
    d = K.shape[-1]

    scale = 1 / math.sqrt(d)

    # Q @ K.T
    # Reads Q and K from HBM and write result to S
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale # [B, H, N, N]

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


torch.manual_seed(123)

q = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
k = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
v = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)

o = naive_attention(q,k,v)
# print(o[:1,:1,:10, :10])
o1 = pytorch_flash_attn2(q,k,v)
# print("Next: ====================")
# print(o1[:1,:1,:10, :10])
assert torch.allclose(o, o1, atol=1e-3), "Outputs do not match"
