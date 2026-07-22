#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

__global__ void fwd_kernel_half(const __half *Q, const __half *K, const __half *V,
                                int B, int H, int N, int d,
                                int Br, int Bc, int Tr, int Tc, float scale,
                                __half *O, float *l, float *m) {

  int batch = blockIdx.x;
  int head = blockIdx.y;
  int tx = threadIdx.x;

  int bh_offset = (batch * H + head) * N * d;
  int stats_offset = (batch * H + head) * N;

  // Shared memory partitioning (Note: Shared memory dynamically allocated as bytes)
  extern __shared__ float sram[];
  __half *Qi = (__half*)sram;                  // Size: Br * d * sizeof(__half)
  __half *Oi = Qi + (Br * d);                 // Size: Br * d * sizeof(__half)
  __half *Kj = Oi + (Br * d);                 // Size: Bc * d * sizeof(__half)
  __half *Vj = Kj + (Bc * d);                 // Size: Bc * d * sizeof(__half)

  // li and mi remain float for numerical stability during online softmax updates
  float *li = (float*)(Vj + (Bc * d));        // Size: Br * sizeof(float)
  float *mi = li + Br;                        // Size: Br * sizeof(float)

  __half h_scale = __float2half(scale);

  // Outer loop: Iterate over K and V tiles
  for (int j = 0; j < Tc; j++) {

    // 1. Collective parallel loading of Kj, Vj to SRAM
    for (int x = tx; x < Bc * d; x += Br) {
        int row = x / d;
        int col = x % d;
        int k_v_row_idx = j * Bc + row;

        if (k_v_row_idx < N) {
            Kj[x] = K[bh_offset + k_v_row_idx * d + col];
            Vj[x] = V[bh_offset + k_v_row_idx * d + col];
        } else {
            Kj[x] = __float2half(0.0f);
            Vj[x] = __float2half(0.0f);
        }
    }
    __syncthreads();

    // Inner loop: Iterate over Q and O tiles
    for (int i = 0; i < Tr; i++) {
      int q_row_idx = i * Br + tx;

      // 2. Load Qi, Oi to SRAM, l and m to shared memory tracking arrays
      if (q_row_idx < N) {
        for (int x = 0; x < d; x++) {
            Qi[(tx * d) + x] = Q[bh_offset + q_row_idx * d + x];
            if (j == 0) {
                Oi[tx * d + x] = __float2half(0.0f);
            } else {
                Oi[tx * d + x] = O[bh_offset + q_row_idx * d + x];
            }
        }
        if (j == 0) {
          mi[tx] = -INFINITY;
          li[tx] = 0.0f;
        } else {
          mi[tx] = m[stats_offset + q_row_idx];
          li[tx] = l[stats_offset + q_row_idx];
        }
      }
      __syncthreads();

      // 3. Compute Attention Sub-matrix (Qi x Kj^T)
      if (q_row_idx < N) {
        float row_m_prev = mi[tx];
        float row_l_prev = li[tx];

        float row_m_new = row_m_prev;
        float S_row[64]; // Static buffer tracking current max block matrix rows

        for (int col_k = 0; col_k < Bc; ++col_k) {
            int k_global_row = j * Bc + col_k;
            float score = 0.0f;

            if (k_global_row < N) {
                // Accumulate in FP32 for statistical stability
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
        row_l_new = (alpha * row_l_prev) + row_l_new;

        // 4. Update Output tile
        for (int col_v = 0; col_v < d; ++col_v) {
            float pv_sum = 0.0f;
            for (int col_k = 0; col_k < Bc; ++col_k) {
                pv_sum += S_row[col_k] * __half2float(Vj[col_k * d + col_v]);
            }

            float updated_val = (alpha * row_l_prev * __half2float(Oi[tx * d + col_v]) + pv_sum)/row_l_new;
            Oi[tx * d + col_v] = __float2half(updated_val);
        }

        mi[tx] = row_m_new;
        li[tx] = row_l_new;

        // Write back updated row to Global Memory (HBM)
        for (int col = 0; col < d; ++col) {
            O[bh_offset + q_row_idx * d + col] = Oi[tx * d + col];
        }
        m[stats_offset + q_row_idx] = row_m_new;
        l[stats_offset + q_row_idx] = row_l_new;
      }
      __syncthreads();
    }
  }
}

torch::Tensor flash_attn_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be CUDA tensors");
  TORCH_CHECK(Q.dim() == 4, "Q must be 4D (B, H, N, D)");
  TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q, K, V shapes must match");

  // Adjusted Type Check: expect float16 / Half precision
  TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "flash_attn_V1 expects float16 inputs");

  Q = Q.contiguous();
  K = K.contiguous();
  V = V.contiguous();

  int batch = Q.size(0);
  int num_heads = Q.size(1);
  int seq_len = Q.size(2);
  int head_dim = Q.size(3);

  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

  int Br = 32;
  int Bc = 32;

  auto O = torch::zeros_like(Q);

  // Note: l and m statistics track row-level exponentials and must remain FP32 to prevent catastrophic underflow/overflow
  auto l = torch::zeros({batch, num_heads, seq_len}, Q.options().dtype(torch::kFloat32));
  auto m = torch::full({batch, num_heads, seq_len}, -INFINITY, Q.options().dtype(torch::kFloat32));

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
        reinterpret_cast<__half *>(O.data_ptr<at::Half>()),
        l.data_ptr<float>(),
        m.data_ptr<float>());

  return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("flash_attn_v1", &flash_attn_kernel, "Flash Attention v1 in CUDA (Float16)");
}
