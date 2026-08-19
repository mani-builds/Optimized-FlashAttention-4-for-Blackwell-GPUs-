#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <mma.h>

using namespace nvcuda;

// FlashAttention-v2 (forward) using wmma tensor-core mma (m16n16k16, fp16 in / fp32 acc).
//
// Grid:  (B, H)  -- one block per (batch, head)
// Block: NWARPS warps (each 32 threads). BR query rows per block tile.
// Warp w owns query rows [warp*16, warp*16+16) of the block tile and, per FA2
// work partitioning, loops over ALL key/value tiles itself.
//
// Algorithm: online softmax with an UNNORMALIZED output accumulator (rescaled by
// alpha = exp(m_prev - m_new) once per K/V block), normalized once at the end
// (O = O / l). All reduction steps go through shared-memory scratch so no manual
// wmma fragment element addressing is required.
template <int BR, int BC, int NWARPS>
__global__ void fwd_kernel_wmma(const __half *Q, const __half *K, const __half *V,
                                int B, int H, int N, int d,
                                int Tr, int Tc, float scale,
                                __half *O) {

  int batch = blockIdx.x;
  int head = blockIdx.y;
  int tid  = threadIdx.x;
  int warp = tid / 32;
  int lane = tid % 32;
  int NT   = NWARPS * 32;

  int bh_offset = (batch * H + head) * N * d;

  // Scratch S tile width: S is BC wide, the PV result is d wide.
  int SSW = (BC > d) ? BC : d;

  // Shared memory partitioning (dynamically allocated as bytes).
  extern __shared__ float sram[];
  __half *Qi        = (__half*)sram;                                  // BR * d
  __half *Kj        = Qi + (BR * d);                                  // BC * d
  __half *Vj        = Kj + (BC * d);                                  // BC * d
  float  *S_scratch = (float*)(Vj + (BC * d));                        // NWARPS*16*SSW (reused as PV temp)
  __half *P_scratch = (__half*)(S_scratch + (NWARPS * 16 * SSW));     // NWARPS*16*BC
  float  *O_scratch = (float*)(P_scratch + (NWARPS * 16 * BC));       // NWARPS*16*d
  float  *m_w       = O_scratch + (NWARPS * 16 * d);                  // NWARPS*16
  float  *l_w       = m_w + (NWARPS * 16);                            // NWARPS*16

  float  *my_S = S_scratch + warp * (16 * SSW);   // 16 x BC fp32 (also PV temp 16 x d)
  __half *my_P = P_scratch + warp * (16 * BC);   // 16 x BC fp16
  float  *my_O = O_scratch + warp * (16 * d);    // 16 x d  fp32
  float  *my_m = m_w + warp * 16;                // 16
  float  *my_l = l_w + warp * 16;                // 16

  const int MAXD16 = 8;  // head_dim <= 128

  for (int i = 0; i < Tr; i++) {

    // 1. Load Qi into SRAM once (zero-pad rows beyond N).
    for (int x = tid; x < (BR * d) / 8; x += NT) {
      int row = (x * 8) / d;
      int col = (x * 8) % d;
      int q_row_idx = i * BR + row;
      if (q_row_idx < N) {
        float4 q_val = *reinterpret_cast<const float4*>(&Q[bh_offset + q_row_idx * d + col]);
        reinterpret_cast<float4*>(Qi)[x] = q_val;
      } else {
        reinterpret_cast<float4*>(Qi)[x] = make_float4(0, 0, 0, 0);
      }
    }

    // 2. Initialize on-chip running statistics and output accumulator.
    int r    = lane % 16;
    int half = lane / 16;
    for (int c = half * (d / 2); c < (half + 1) * (d / 2); c++) {
      my_O[r * d + c] = 0.0f;
    }
    if (lane < 16) {
      my_m[lane] = -INFINITY;
      my_l[lane] = 0.0f;
    }
    __syncthreads();

    // Inner loop: iterate over K/V tiles (columns of the attention matrix).
    for (int j = 0; j < Tc; j++) {

      // 3. Load Kj, Vj into SRAM (zero-pad rows beyond N).
      for (int x = tid; x < (BC * d) / 8; x += NT) {
        int row = (x * 8) / d;
        int col = (x * 8) % d;
        int k_v_row_idx = j * BC + row;
        if (k_v_row_idx < N) {
          float4 k_val = *reinterpret_cast<const float4*>(&K[bh_offset + k_v_row_idx * d + col]);
          float4 v_val = *reinterpret_cast<const float4*>(&V[bh_offset + k_v_row_idx * d + col]);
          reinterpret_cast<float4*>(Kj)[x] = k_val;
          reinterpret_cast<float4*>(Vj)[x] = v_val;
        } else {
          reinterpret_cast<float4*>(Kj)[x] = make_float4(0, 0, 0, 0);
          reinterpret_cast<float4*>(Vj)[x] = make_float4(0, 0, 0, 0);
        }
      }
      __syncthreads();

      // 4. S = scale * Qi_warp * Kj^T  via wmma (m16n16k16, k = d, n = BC).
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[BC / 16];
      #pragma unroll
      for (int cb = 0; cb < BC / 16; cb++) wmma::fill_fragment(s_frag[cb], 0.0f);
      for (int kk = 0; kk < d; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, &Qi[warp * 16 * d + kk], d);
        #pragma unroll
        for (int cb = 0; cb < BC / 16; cb++) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
          wmma::load_matrix_sync(b_frag, &Kj[cb * 16 * d + kk], d);
          wmma::mma_sync(s_frag[cb], a_frag, b_frag, s_frag[cb]);
        }
      }
      #pragma unroll
      for (int cb = 0; cb < BC / 16; cb++)
        for (int e = 0; e < s_frag[cb].num_elements; e++) s_frag[cb].x[e] *= scale;

      // 5. Store S to per-warp SRAM scratch.
      #pragma unroll
      for (int cb = 0; cb < BC / 16; cb++)
        wmma::store_matrix_sync(&my_S[cb * 16], s_frag[cb], BC, wmma::mem_row_major);
      __syncwarp();

      // 6. Online softmax (row max / alpha / row sum) with padding-aware masks.
      int cstart = half * (BC / 2);
      float rowmax = -INFINITY;
      for (int c = cstart; c < cstart + BC / 2; c++) {
        int gcol = j * BC + c;
        if (gcol < N) {
          float sc = my_S[r * BC + c];
          if (sc > rowmax) rowmax = sc;
        }
      }
      rowmax = fmaxf(rowmax, __shfl_xor_sync(0xffffffff, rowmax, 16));

      float m_prev = my_m[r];
      float m_new  = fmaxf(m_prev, rowmax);
      float a_row  = __expf(m_prev - m_new);

      float row_l = 0.0f;
      for (int c = cstart; c < cstart + BC / 2; c++) {
        int gcol = j * BC + c;
        float pval = (gcol < N) ? __expf(my_S[r * BC + c] - m_new) : 0.0f;
        my_P[r * BC + c] = __float2half(pval);
        row_l += pval;
      }
      row_l += __shfl_xor_sync(0xffffffff, row_l, 16);
      my_l[r] = a_row * my_l[r] + row_l;
      my_m[r] = m_new;
      __syncwarp();

      // 7. O += P * V  via wmma (k = BC, n = d), result into my_S (PV temp).
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[MAXD16];
      #pragma unroll
      for (int nd = 0; nd < MAXD16; nd++) wmma::fill_fragment(o_frag[nd], 0.0f);
      for (int kb = 0; kb < BC; kb += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa_frag;
        wmma::load_matrix_sync(pa_frag, &my_P[kb], BC);
        for (int nd = 0; nd < d / 16; nd++) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
          wmma::load_matrix_sync(v_frag, &Vj[kb * d + nd * 16], d);
          wmma::mma_sync(o_frag[nd], pa_frag, v_frag, o_frag[nd]);
        }
      }
      for (int nd = 0; nd < d / 16; nd++)
        wmma::store_matrix_sync(&my_S[nd * 16], o_frag[nd], d, wmma::mem_row_major);
      __syncwarp();

      // 8. my_O = alpha * my_O + (P*V), entirely in SRAM (row r, half of d cols).
      for (int c = half * (d / 2); c < (half + 1) * (d / 2); c++) {
        my_O[r * d + c] = a_row * my_O[r * d + c] + my_S[r * d + c];
      }
      __syncwarp();
    }

    // 9. Final normalize O = O / l and write to HBM (skip rows beyond N).
    int q_row_idx = i * BR + warp * 16 + r;
    if (q_row_idx < N) {
      float inv_l = 1.0f / my_l[r];
      for (int c = half * (d / 2); c < (half + 1) * (d / 2); c++) {
        O[bh_offset + q_row_idx * d + c] = __float2half(my_O[r * d + c] * inv_l);
      }
    }
  }
}

// SRAM usage for a given config (bytes).
static inline int wmma_sram_size(int BR, int BC, int NWARPS, int d) {
  int SSW = (BC > d) ? BC : d;
  return (BR * d + 2 * BC * d) * (int)sizeof(__half)
       + (NWARPS * 16 * SSW) * (int)sizeof(float)   // S scratch (reused as PV temp)
       + (NWARPS * 16 * BC)  * (int)sizeof(__half)   // P scratch
       + (NWARPS * 16 * d)   * (int)sizeof(float)    // O scratch
       + (NWARPS * 16 * 2)   * (int)sizeof(float);   // m, l
}

torch::Tensor flash_attn_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be CUDA tensors");
  TORCH_CHECK(Q.dim() == 4, "Q must be 4D (B, H, N, D)");
  TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q, K, V shapes must match");
  TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "flash_attn_v2_mma expects float16 inputs");

  Q = Q.contiguous();
  K = K.contiguous();
  V = V.contiguous();

  int batch = Q.size(0);
  int num_heads = Q.size(1);
  int seq_len = Q.size(2);
  int head_dim = Q.size(3);
  TORCH_CHECK(head_dim % 16 == 0, "head_dim must be a multiple of 16 for m16n16k16 wmma");
  TORCH_CHECK(head_dim <= 128, "head_dim must be <= 128 for the wmma kernel");

  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

  int Tr = (seq_len + 63) / 64;   // filled below per chosen config
  int Tc = (seq_len + 63) / 64;

  auto O = torch::zeros_like(Q);
  float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  dim3 grid(batch, num_heads);

  // Pick the largest tile config that fits in the default 48KB dynamic SRAM limit.
  bool launched = false;
  if (wmma_sram_size(64, 64, 4, head_dim) <= max_sram_size) {
    dim3 block(4 * 32);
    Tr = (seq_len + 63) / 64;
    Tc = (seq_len + 63) / 64;
    fwd_kernel_wmma<64, 64, 4><<<grid, block, wmma_sram_size(64, 64, 4, head_dim)>>>(
        reinterpret_cast<__half *>(Q.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(K.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(V.data_ptr<at::Half>()),
        batch, num_heads, seq_len, head_dim, Tr, Tc, softmax_scale,
        reinterpret_cast<__half *>(O.data_ptr<at::Half>()));
    launched = true;
  } else if (wmma_sram_size(32, 64, 2, head_dim) <= max_sram_size) {
    dim3 block(2 * 32);
    Tr = (seq_len + 31) / 32;
    Tc = (seq_len + 63) / 64;
    fwd_kernel_wmma<32, 64, 2><<<grid, block, wmma_sram_size(32, 64, 2, head_dim)>>>(
        reinterpret_cast<__half *>(Q.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(K.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(V.data_ptr<at::Half>()),
        batch, num_heads, seq_len, head_dim, Tr, Tc, softmax_scale,
        reinterpret_cast<__half *>(O.data_ptr<at::Half>()));
    launched = true;
  } else if (wmma_sram_size(32, 32, 2, head_dim) <= max_sram_size) {
    dim3 block(2 * 32);
    Tr = (seq_len + 31) / 32;
    Tc = (seq_len + 31) / 32;
    fwd_kernel_wmma<32, 32, 2><<<grid, block, wmma_sram_size(32, 32, 2, head_dim)>>>(
        reinterpret_cast<__half *>(Q.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(K.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(V.data_ptr<at::Half>()),
        batch, num_heads, seq_len, head_dim, Tr, Tc, softmax_scale,
        reinterpret_cast<__half *>(O.data_ptr<at::Half>()));
    launched = true;
  } else if (wmma_sram_size(16, 16, 1, head_dim) <= max_sram_size) {
    dim3 block(1 * 32);
    Tr = (seq_len + 15) / 16;
    Tc = (seq_len + 15) / 16;
    fwd_kernel_wmma<16, 16, 1><<<grid, block, wmma_sram_size(16, 16, 1, head_dim)>>>(
        reinterpret_cast<__half *>(Q.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(K.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(V.data_ptr<at::Half>()),
        batch, num_heads, seq_len, head_dim, Tr, Tc, softmax_scale,
        reinterpret_cast<__half *>(O.data_ptr<at::Half>()));
    launched = true;
  }

  TORCH_CHECK(launched, "No wmma tile config fits in available shared memory");
  return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("flash_attn_v2_mma", &flash_attn_kernel, "Flash Attention V2 with wmma tensor cores (Float16)");
}