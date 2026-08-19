#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void fwd_kernel_half(const __half *Q, const __half *K, const __half *V,
                                int B, int H, int N, int d,
                                int Br, int Bc, int Tr, int Tc, float scale,
                                __half *O) {

  // Map threads
  int batch = blockIdx.x;
  int head = blockIdx.y;
  int tx = threadIdx.x;

  int bh_offset = (batch * H + head) * N * d;

  // Shared memory partitioning (Note: Shared memory dynamically allocated as bytes)
  extern __shared__ float sram[];
  __half *Qi = (__half*)sram;                  // Size: Br * d * sizeof(__half)
  __half *Oi = Qi + (Br * d);                 // Size: Br * d * sizeof(__half)
  __half *Kj = Oi + (Br * d);                 // Size: Bc * d * sizeof(__half)
  __half *Vj = Kj + (Bc * d);                 // Size: Bc * d * sizeof(__half)

  // li and mi remain float for numerical stability during online softmax updates
  float *li = (float*)(Vj + (Bc * d));        // Size: Br * sizeof(float)
  float *mi = li + Br;                        // Size: Br * sizeof(float)

  // Outer loop: Iterate over Q and O tiles (query blocks)
  for (int i = 0; i < Tr; i++) {
    int q_row_idx = i * Br + tx;

    // 1. Load Qi into SRAM once; initialize Oi=0, li=0, mi=-inf on-chip
    if (q_row_idx < N) {
      for (int x = 0; x < d / 8; x++) {
        float4 q_val = *reinterpret_cast<const float4*>(&Q[bh_offset + q_row_idx * d + x * 8]);
        reinterpret_cast<float4*>(Qi)[(tx * d / 8) + x] = q_val;
        reinterpret_cast<float4*>(Oi)[(tx * d / 8) + x] = make_float4(0,0,0,0);
      }
      mi[tx] = -INFINITY;
      li[tx] = 0.0f;
    }
    __syncthreads();

    // Inner loop: Iterate over K and V tiles
    for (int j = 0; j < Tc; j++) {

      // 2. Collective parallel loading of Kj, Vj to SRAM
      for (int x = tx; x < (Bc * d) / 8; x += Br) {
        int offset = x * 8;
        int row = offset / d;
        int col = offset % d;
        int k_v_row_idx = j * Bc + row;

        if (k_v_row_idx < N) {
          // Each vectorized float4 loads 4 FP32 values, for __half FP16, it loads 8
          float4 k_val = *reinterpret_cast<const float4*>(&K[bh_offset + k_v_row_idx * d + col]);
          float4 v_val = *reinterpret_cast<const float4*>(&V[bh_offset + k_v_row_idx * d + col]);
          reinterpret_cast<float4*>(Kj)[x] = k_val;
          reinterpret_cast<float4*>(Vj)[x] = v_val;
        } else {
          // while make_float4 take four arguments(coz it works on FP32), it correctly sets
          // the entire 128-bit (4*32) memory block to zero, which covers 8 __half elements
          reinterpret_cast<float4*>(Kj)[x] = make_float4(0,0,0,0);
          reinterpret_cast<float4*>(Vj)[x] = make_float4(0,0,0,0);
        }
      }
      __syncthreads();

      // 3. Compute Attention Sub-matrix (Qi x Kj^T)
      if (q_row_idx < N) {
        float row_m_prev = mi[tx];
        float row_l_prev = li[tx];

        float row_m_new = row_m_prev;
        float S_row[64]; // Static buffer tracking current max block matrix rows

        #pragma unroll
        for (int col_k = 0; col_k < Bc; ++col_k) {
            int k_global_row = j * Bc + col_k;
            float score = 0.0f;

            if (k_global_row < N) {
                // Accumulate in FP32 for statistical stability
                #pragma unroll
                for (int k = 0; k < d; ++k) {
                    score += __half2float(Qi[tx * d + k]) * __half2float(Kj[col_k * d + k]);
                }
                score *= scale;
            } else {
                score = -INFINITY;
            }
            S_row[col_k] = score;
            if (score > row_m_new) row_m_new = score;
        }

        // Compute updates according to Online Softmax algorithm
        float row_l_new = 0.0f;
        for (int col_k = 0; col_k < Bc; ++col_k) {
            if ((j * Bc + col_k) < N) {
                S_row[col_k] = expf(S_row[col_k] - row_m_new);
                row_l_new += S_row[col_k];
            } else {
                S_row[col_k] = 0.0f;
            }
        }
        float alpha = expf(row_m_prev - row_m_new);

        // 4. FlashAttention-v2 update: accumulate O UNNORMALIZED (raw running
        //    sum), rescaling by alpha only once per block. The final division
        //    O = O / l happens once, after the K/V loop completes.
        #pragma unroll
        for (int col_v = 0; col_v < d; ++col_v) {
            float pv_sum = 0.0f;
            #pragma unroll
            for (int col_k = 0; col_k < Bc; ++col_k) {
                pv_sum += S_row[col_k] * __half2float(Vj[col_k * d + col_v]);
            }

            float acc = (alpha * __half2float(Oi[tx * d + col_v])) + pv_sum;
            Oi[tx * d + col_v] = __float2half(acc);
        }

        li[tx] = (alpha * row_l_prev) + row_l_new;
        mi[tx] = row_m_new;
      }
      __syncthreads();
    }

    // 5. Single final normalization O = O / l, then write back to HBM once
    if (q_row_idx < N) {
      float row_l_final = li[tx];
      for (int col_v = 0; col_v < d; ++col_v) {
        Oi[tx * d + col_v] = __float2half(__half2float(Oi[tx * d + col_v]) / row_l_final);
      }
      for (int col_v_8 = 0; col_v_8 < d / 8; ++col_v_8) {
        reinterpret_cast<float4*>(&O[bh_offset + q_row_idx * d + col_v_8 * 8])[0] =
            reinterpret_cast<float4*>(Oi)[tx * (d / 8) + col_v_8];
      }
    }
  }
}

torch::Tensor flash_attn_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be CUDA tensors");
  TORCH_CHECK(Q.dim() == 4, "Q must be 4D (B, H, N, D)");
  TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q, K, V shapes must match");

  // Adjusted Type Check: expect float16 / Half precision
  TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "flash_attn_V2 expects float16 inputs");

  Q = Q.contiguous();
  K = K.contiguous();
  V = V.contiguous();

  int batch = Q.size(0);
  int num_heads = Q.size(1);
  int seq_len = Q.size(2);
  int head_dim = Q.size(3);
  TORCH_CHECK(head_dim % 8 == 0, "head_dim must be a multiple of 8 for vectorized loads");

  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

  int Br = 128;
  int Bc = 32;

  auto O = torch::zeros_like(Q);

  int Tr = (seq_len + Br - 1) / Br;
  int Tc = (seq_len + Bc - 1) / Bc;

  dim3 threadsPerBlock(Br);
  dim3 blocksPerGrid(batch, num_heads);

  // Calculate SRAM bytes required:
  // Q, O, K, V use half (2 bytes). l, m use float (4 bytes).
  int sram_size = (2 * Br * head_dim * sizeof(__half))  // Qi, Oi
                + (2 * Bc * head_dim * sizeof(__half))  // Kj, Vj
                + (2 * Br * sizeof(float));             // li, mi

  if (sram_size > max_sram_size) {
    Br = 16; Bc = 16;
    sram_size = (2 * Br * head_dim * sizeof(__half))
              + (2 * Bc * head_dim * sizeof(__half))
              + (2 * Br * sizeof(float));
  }

  float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  fwd_kernel_half<<<blocksPerGrid, threadsPerBlock, sram_size>>>(
        reinterpret_cast<__half *>(Q.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(K.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(V.data_ptr<at::Half>()),
        batch, num_heads, seq_len, head_dim, Br, Bc, Tr, Tc,
        softmax_scale,
        reinterpret_cast<__half *>(O.data_ptr<at::Half>()));

  return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("flash_attn_v2", &flash_attn_kernel, "Flash Attention V2 in CUDA (Float16)");
}