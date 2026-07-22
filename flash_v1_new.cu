#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <math.h>
#include <torch/extension.h>

__global__ void fwd_kernel(float *Q, float *K, float *V, int B, int H, int N,
                           int d, int Br, int Bc, int Tr, int Tc, float scale,
                           float *O,
                           float *l, float *m
                          ) {

  // Which (batch, head) slice does this block own?
  int batch = blockIdx.x;
  int head = blockIdx.y;

  // Map threads to query row index
  // Thread index within the block:
  // Each thread handles a single query row within the current Q tile.
  // Even though we launch only 'Br' threads, and calculating a Br x d in
  // a single snapshot, the nested loops make sure that these 'Br' threads
  // cover the entire N. (0, 1, ..., Tr)
  int tx = threadIdx.x; // Single dimension


  // base offsets for this batch and head
  int bh_offset = (batch * H + head) * N * d;
  int stats_offset = (batch * H + head) * N;

  // Shared memory partitioning
  extern __shared__ float sram[];
  float *Qi = sram;                        // Size: Br * d
  float *Oi = Qi + (Br * d);               // Size: Br * d
  float *li = Oi + (Br * d);               // Size: Br
  float *mi = li + Br;                     // Size: Br
  float *Kj = mi + Br;                     // Size: Bc * d
  float *Vj = Kj + (Bc * d);               // Size: Bc * d

  // Outer loop: Iterate over K and V tiles
  for (int j = 0; j < Tc; j++) {
    //1. Load Kj, Vj to SRAM
    for (int x = tx; x < Bc * d; x += Br) {
        int row = x / d;
        int col = x % d;
        int k_v_row_idx = j * Bc + row;

        if (k_v_row_idx < N) {
            Kj[x] = K[bh_offset + k_v_row_idx * d + col];
            Vj[x] = V[bh_offset + k_v_row_idx * d + col];
        } else {
            Kj[x] = 0.0f;
            Vj[x] = 0.0f;
    }
}    __syncthreads(); //to make sure Kj and Vj are fully loaded before entering inner loop

    // Inner loop: Iterate over Q and O (and therefore output tiles)
    for (int i = 0; i < Tr; i++) {
      int q_row_idx = i * Br + tx;

      //2. Load Qi, Oi to SRAM, l and m to registers
      if (q_row_idx < N){
        for (int x = 0; x < d; x++) {
            Qi[(tx * d) + x] = Q[bh_offset + q_row_idx * d + x];
            // If this is the first outer-loop step, initialize Oi to 0
            if (j == 0) {
                Oi[tx * d + x] = 0.0f;
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
      // 3. Compute Attention Sub-matrix (Qi x Kj^T) and update statistics
    if (q_row_idx < N) {
        float row_m_prev = mi[tx];
        float row_l_prev = li[tx];

        // Find local max row value for stable softmax
        float row_m_new = row_m_prev;

        // Array to hold intermediate un-normalized attention scores (S) for this thread's row
        // Max size depends on Bc limit (e.g., 32 or 64)
        float S_row[64];

        for (int col_k = 0; col_k < Bc; ++col_k) {
            int k_global_row = j * Bc + col_k;
            float score = 0.0f;

            if (k_global_row < N) {
                for (int k = 0; k < d; ++k) {
                    score += Qi[tx * d + k] * Kj[col_k * d + k];
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

        // 4. Update Output tile Oi = (alpha * row_l_prev * Oi + P_ij * Vj) / row_l_new
        for (int col_v = 0; col_v < d; ++col_v) {
            float pv_sum = 0.0f;
            for (int col_k = 0; col_k < Bc; ++col_k) {
                pv_sum += S_row[col_k] * Vj[col_k * d + col_v];
            }
            Oi[tx * d + col_v] = (alpha * row_l_prev * Oi[tx * d + col_v] + pv_sum);
        }

        // Save statistics back to shared memory registers
        mi[tx] = row_m_new;
        li[tx] = row_l_new;

        // Write back updated row to Global Memory (HBM)
        for (int col = 0; col < d; ++col) {
            O[bh_offset + q_row_idx * d + col] = Oi[tx * d + col];
        }
        m[stats_offset + q_row_idx] = row_m_new;
        l[stats_offset + q_row_idx] = row_l_new;
        }
        // Sync before moving to the next Q tile row iteration
        __syncthreads();
    }
  }
}


torch::Tensor flash_attn_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be CUDA tensors");
  TORCH_CHECK(Q.dim() == 4, "Q must be 4D (B, H, N, D)");
  TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q, K, V shapes must match");
  TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "flash_attn_V1 expects float32 inputs");

  Q = Q.contiguous();
  K = K.contiguous();
  V = V.contiguous();

  int batch, num_heads, seq_len, head_dim;
  batch = Q.size(0); num_heads = Q.size(1); seq_len = Q.size(2); head_dim = Q.size(3);


  // Q.shape = [B, H, N, d]

  // SRAM
  int max_sram_size; // M
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

  // Block sizes
  int Br = 32;
  int Bc = 32;

  // Initialize O (output), l and m (intermediate statistics) in HBM
  auto O = torch::zeros_like(Q); // N x d
  auto l = torch::zeros({batch, num_heads, seq_len}, Q.options()); // N
  auto m = torch::full({batch, num_heads, seq_len}, -INFINITY, Q.options()); // N

  // TILE sizes
  // Q is divided into Tr tiles each of size Br x d
  // K is divided into Tc tiles each of size Bc x d
  // V is divided into Tc tiles each of size Bc x d

  // O is divided into Tr tiles each of size Br x d
  // l is divided into Tr blocks each of size Br
  // m is divided into Tr blocks each of size Br
  int Tr = (seq_len + Br - 1) / Br ;// no. of tiles of Row
  int Tc = (seq_len + Bc - 1) / Bc; // ceil division

  // Launch parameters
  // We launch (batch, num_heads) no. of blocks i.e each block owns ONE (batch, head) slice
  // So each thread block performs complete(fused) attention calulation of O of size (N,d)
  // This way we technically launched a 4D tensor operation (B, H, N, d)
  // Each thread block handles a single Q tile
  dim3 threadsPerBlock(Br);
  dim3 blocksPerGrid(batch, num_heads);

  // SRAM per block needed for Qi, Oi, li, mi and Ki, Vi
  int sram_size = (2 * Br * head_dim * sizeof(float)) + (2 * Br * sizeof(float))
                  + (2 * Bc * head_dim * sizeof(float));

  // Ensure it fits in SRAM
  if (sram_size > max_sram_size) {
    // Fallback or scale down if head_dim is massive
    Br = 16; Bc = 16;
    sram_size = (2 * Br * head_dim * sizeof(float)) + (2 * Br * sizeof(float))
                + (2 * Bc * head_dim * sizeof(float));
  }
  printf("\nMax shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

  float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  fwd_kernel<<<blocksPerGrid, threadsPerBlock, sram_size>>>(
        reinterpret_cast<float *>(Q.data_ptr()),
        reinterpret_cast<float *>(K.data_ptr()),
        reinterpret_cast<float *>(V.data_ptr()), batch, num_heads, seq_len,
                                                head_dim, Br, Bc, Tr, Tc,
        softmax_scale,
        reinterpret_cast<float *>(O.data_ptr()),
        reinterpret_cast<float *>(l.data_ptr()),
        reinterpret_cast<float *>(m.data_ptr()));

  return O;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("flash_attn_v1", &flash_attn_kernel, "Flash Attension v1 in CUDA");
 }
