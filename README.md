Optimization of FlashAttention-4 on Blackwell GPUs for LLM inference

Implementaion details of FlashAttention-4 from scratch in raw CUDA (C++ kernels + PyTorch custom operator), moving beyond FlashAttention-v1 techniques to the latest Hopper/Blackwell-aware methods: warp-specialization + asynchronous TMA (Tensor Memory Accelerator) loads/stores, interleaved block-wise GEMM and softmax, incoherent processing + block quantization for FP8 (and early FP4 exploration), and CuTe-style tensor expressions ported to raw CUDA for maximum control.


Leveraged Blackwell-specific hardware: native 5th-gen Tensor Cores with massive FP8/FP6/FP4 throughput, increased register/shared memory bandwidth, and the latest async copy primitives (CUDA 12.8+ with sm_100/sm_12x compute capability).


Integrated as a drop-in torch.nn.functional replacement via torch.utils.cpp_extension (or CUTLASS 3.x helpers if you want to hybridize later) so it works seamlessly with Llama-3.1, Mistral, or GPT-style models in both training and inference.


Benchmarked vanilla PyTorch scaled_dot_product_attention vs. optimized kernel vs. official flash-attn-4 on Blackwell B100 GPUs (use cloud like RunPod, Lambda, or CoreWeave if you don’t have on-prem access). Tested forward + backward passes on sequence lengths 512 → 8K+ tokens, batch sizes typical for inference (1-32), using Nsight Compute, Nsight Systems, and PyTorch Profiler. Target metrics: 1.8–2.4× speedup over FA-2, >70% of Blackwell theoretical peak TFLOPS (aim for 1.5k+ TFLOPS in FP8), and ~2× lower HBM traffic / VRAM usage compared to baseline.
