#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <mma.h>

using namespace nvcuda;

template <int BR, int BC, int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32, 8)
fwd_kernel_wmma(const __half *Q, const __half *K, const __half *V,
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

  extern __shared__ float sram[];
  __half *Qi = (__half*)sram;              // BR * d
  __half *Kj = Qi + (BR * d);              // BC * d
  __half *Vj = Kj + (BC * d);              // BC * d
  __half *Ohalf = Vj + (BC * d);           // NWARPS * 16 * d

  // const int QUAD = lane >> 2;              // 0..7
  // const bool ROWA = true;

  for (int i = 0; i < Tr; i++) {
    __syncthreads();

    for (int x = tid; x < (BR * d) / 8; x += NT) {
      int row = (x * 8) / d;
      int col = (x * 8) % d;
      int q_row_idx = i * BR + row;
      if (q_row_idx < N) {
        reinterpret_cast<float4*>(Qi)[x] =
            *reinterpret_cast<const float4*>(&Q[bh_offset + q_row_idx * d + col]);
      } else {
        reinterpret_cast<float4*>(Qi)[x] = make_float4(0, 0, 0, 0);
      }
    }
    __syncthreads();

    const int MAXD16 = 8;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[MAXD16];
    #pragma unroll
    for (int kk = 0; kk < d; kk += 16)
      wmma::load_matrix_sync(a_frag[kk / 16], &Qi[warp * 16 * d + kk], d);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[MAXD16];
    #pragma unroll
    for (int nd = 0; nd < d / 16; nd++) wmma::fill_fragment(o_frag[nd], 0.0f);

    float mrowA = -INFINITY, mrowB = -INFINITY;
    float lrowA = 0.0f, lrowB = 0.0f;

    for (int j = 0; j < Tc; j++) {

      for (int x = tid; x < (BC * d) / 8; x += NT) {
        int row = (x * 8) / d;
        int col = (x * 8) % d;
        int k_v_row_idx = j * BC + row;
        if (k_v_row_idx < N) {
          reinterpret_cast<float4*>(Kj)[x] =
              *reinterpret_cast<const float4*>(&K[bh_offset + k_v_row_idx * d + col]);
          reinterpret_cast<float4*>(Vj)[x] =
              *reinterpret_cast<const float4*>(&V[bh_offset + k_v_row_idx * d + col]);
        } else {
          reinterpret_cast<float4*>(Kj)[x] = make_float4(0, 0, 0, 0);
          reinterpret_cast<float4*>(Vj)[x] = make_float4(0, 0, 0, 0);
        }
      }
      __syncthreads();

      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[BC / 16];
      #pragma unroll
      for (int cb = 0; cb < BC / 16; cb++) wmma::fill_fragment(s_frag[cb], 0.0f);

      #pragma unroll
      for (int kk = 0; kk < d; kk += 16) {
        #pragma unroll
        for (int cb = 0; cb < BC / 16; cb++) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
          wmma::load_matrix_sync(b_frag, &Kj[cb * 16 * d + kk], d);
          wmma::mma_sync(s_frag[cb], a_frag[kk / 16], b_frag, s_frag[cb]);
        }
      }

      int nume = s_frag[0].num_elements;

      float rowmaxB = -INFINITY, rowmaxA = -INFINITY;
      #pragma unroll
      for (int cb = 0; cb < BC / 16; cb++) {
        int cbase = j * BC + cb * 16;
        #pragma unroll
        for (int e = 0; e < nume; e++) {
          float s = s_frag[cb].x[e] * scale;
          s_frag[cb].x[e] = s;
          if (cbase + ((lane & 3) << 1) + (e & 1) + 8 * ((e >> 2) & 1) < N) {
            float r = (lane >> 2) + 8 * ((e >> 1) & 1);
            if (r < 8) rowmaxA = fmaxf(rowmaxA, s);
            else       rowmaxB = fmaxf(rowmaxB, s);
          }
        }
      }
      rowmaxA = fmaxf(rowmaxA, __shfl_xor_sync(0xffffffffu, rowmaxA, 1));
      rowmaxA = fmaxf(rowmaxA, __shfl_xor_sync(0xffffffffu, rowmaxA, 2));
      rowmaxB = fmaxf(rowmaxB, __shfl_xor_sync(0xffffffffu, rowmaxB, 1));
      rowmaxB = fmaxf(rowmaxB, __shfl_xor_sync(0xffffffffu, rowmaxB, 2));

      float m_newA = fmaxf(mrowA, rowmaxA);
      float m_newB = fmaxf(mrowB, rowmaxB);
      float alphaA = expf(mrowA - m_newA);
      float alphaB = expf(mrowB - m_newB);

      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa_frag[BC / 16];
      float lA = 0.0f, lB = 0.0f;
      #pragma unroll
      for (int cb = 0; cb < BC / 16; cb++) {
        int cbase = j * BC + cb * 16;
        #pragma unroll
        for (int e = 0; e < nume; e++) {
          float s = s_frag[cb].x[e];
          int col = cbase + ((lane & 3) << 1) + (e & 1) + 8 * ((e >> 2) & 1);
          float r = (lane >> 2) + 8 * ((e >> 1) & 1);
          float p = (col < N) ? expf(s - ((r < 8) ? m_newA : m_newB)) : 0.0f;
          pa_frag[cb].x[e] = __float2half(p);
          if (r < 8) lA += p; else lB += p;
        }
      }
      lA += __shfl_xor_sync(0xffffffffu, lA, 1);
      lA += __shfl_xor_sync(0xffffffffu, lA, 2);
      lB += __shfl_xor_sync(0xffffffffu, lB, 1);
      lB += __shfl_xor_sync(0xffffffffu, lB, 2);
      lrowA = alphaA * lrowA + lA;
      lrowB = alphaB * lrowB + lB;
      mrowA = m_newA;
      mrowB = m_newB;

      #pragma unroll
      for (int nd = 0; nd < d / 16; nd++) {
        #pragma unroll
        for (int e = 0; e < nume; e++) {
          float r = (lane >> 2) + 8 * ((e >> 1) & 1);
          o_frag[nd].x[e] *= (r < 8) ? alphaA : alphaB;
        }
      }

      #pragma unroll
      for (int kb = 0; kb < BC / 16; kb++) {
        #pragma unroll
        for (int nd = 0; nd < d / 16; nd++) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
          wmma::load_matrix_sync(v_frag, &Vj[kb * 16 * d + nd * 16], d);
          wmma::mma_sync(o_frag[nd], pa_frag[kb], v_frag, o_frag[nd]);
        }
      }
    }

    float inv_lA = 1.0f / lrowA;
    float inv_lB = 1.0f / lrowB;
    int wbase = warp * 16 * d;
    #pragma unroll
    for (int nd = 0; nd < d / 16; nd++) {
      #pragma unroll
      for (int e = 0; e < o_frag[nd].num_elements; e++) {
        int r = (lane >> 2) + 8 * ((e >> 1) & 1);
        int c = ((lane & 3) << 1) + (e & 1) + 8 * ((e >> 2) & 1);
        int qrow = i * BR + warp * 16 + r;
        float v = (qrow < N) ? o_frag[nd].x[e] * ((r < 8) ? inv_lA : inv_lB) : 0.0f;
        Ohalf[wbase + r * d + nd * 16 + c] = __float2half(v);
      }
    }
    __syncwarp();
    for (int x = lane; x < (16 * d) / 8; x += 32) {
      int row = (x * 8) / d;
      int col = (x * 8) % d;
      int qrow = i * BR + warp * 16 + row;
      if (qrow < N)
        reinterpret_cast<float4*>(&O[bh_offset + qrow * d + col])[0] =
            *reinterpret_cast<const float4*>(&Ohalf[wbase + row * d + col]);
    }
  }
}

static inline int wmma_sram_size(int BR, int BC, int NWARPS, int d) {
  return (BR * d + 2 * BC * d) * (int)sizeof(__half)
       + (NWARPS * 16 * d) * (int)sizeof(__half);
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

  int Tr = (seq_len + 63) / 64;
  int Tc = (seq_len + 63) / 64;

  auto O = torch::zeros_like(Q);
  float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  dim3 grid(batch, num_heads);

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
