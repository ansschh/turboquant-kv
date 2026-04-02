/*
 * rotate_fused.cu — Fused rotation + quantization kernel.
 *
 * Input:  float16/bfloat16 vectors [N, d], rotation [d, d], boundaries [2^b-1], bit_width
 * Output: packed codes [N, ceil(d*b/8)] uint8, norms [N] float32
 *
 * Steps: norm -> normalize -> rotate (shared-mem dot products) -> searchsorted -> bit-pack
 *
 * Each block processes one vector.  Two-phase approach:
 *   Phase 1: compute quantization codes into shared memory
 *   Phase 2: bit-pack from shared memory into global output (no atomics needed)
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// -------------------------------------------------------------------
// Kernel: fused normalize + rotate + quantize + pack
// -------------------------------------------------------------------

__global__ void rotate_and_quantize_kernel(
    const float* __restrict__ vectors,    // [N, d]
    const float* __restrict__ rotation,   // [d, d]
    const float* __restrict__ boundaries, // [n_levels - 1]
    uint8_t* __restrict__ packed_out,     // [N, packed_dim]
    float* __restrict__ norms_out,        // [N]
    int N, int d,
    int bit_width,
    int n_boundaries,
    int packed_dim
) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= N) return;

    // Dynamic shared memory layout:
    //   [0          .. d)             : unit vector (float) / then reused as codes (uint8 cast)
    //   [d          .. d+n_boundaries): boundaries (float)
    //   [d+n_boundaries .. d+n_boundaries+d): codes buffer (uint8, packed into floats)
    extern __shared__ char smem_raw[];
    float* s_unit = (float*)smem_raw;
    float* s_boundaries = s_unit + d;
    uint8_t* s_codes = (uint8_t*)(s_boundaries + n_boundaries);

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Load boundaries into shared memory
    for (int i = tid; i < n_boundaries; i += block_size) {
        s_boundaries[i] = boundaries[i];
    }

    // Step 1: compute norm
    const float* vec = vectors + (size_t)vec_idx * d;
    float local_sum = 0.0f;
    for (int i = tid; i < d; i += block_size) {
        float v = vec[i];
        local_sum += v * v;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

    __shared__ float warp_sums[32];
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    float norm_sq = 0.0f;
    if (tid == 0) {
        int n_warps = (block_size + warpSize - 1) / warpSize;
        for (int i = 0; i < n_warps; i++) norm_sq += warp_sums[i];
        warp_sums[0] = norm_sq;
        norms_out[vec_idx] = sqrtf(norm_sq);
    }
    __syncthreads();
    norm_sq = warp_sums[0];
    float inv_norm = (norm_sq > 1e-20f) ? rsqrtf(norm_sq) : 0.0f;

    // Step 2: normalize and store unit vector in shared memory
    for (int i = tid; i < d; i += block_size) {
        s_unit[i] = vec[i] * inv_norm;
    }
    __syncthreads();

    // Phase 1: rotate and quantize -> store codes in s_codes
    for (int j = tid; j < d; j += block_size) {
        // rotated[j] = sum_k unit[k] * rotation[j][k]  (unit @ rotation^T)
        float val = 0.0f;
        const float* rot_row = rotation + (size_t)j * d;
        for (int k = 0; k < d; k++) {
            val += s_unit[k] * rot_row[k];
        }

        // Searchsorted: boundaries are sorted ascending
        int code = 0;
        for (int b = 0; b < n_boundaries; b++) {
            if (val > s_boundaries[b]) code = b + 1;
            else break;
        }

        s_codes[j] = (uint8_t)code;
    }
    __syncthreads();

    // Phase 2: bit-pack from s_codes into global memory
    // Each thread handles a subset of output bytes to avoid conflicts.
    // Packing layout: bit plane i occupies bytes [i*bpp .. (i+1)*bpp - 1]
    // Within a plane, byte b packs dimensions [8*b .. 8*b+7].
    int bytes_per_plane = (d + 7) / 8;
    uint8_t* packed_row = packed_out + (size_t)vec_idx * packed_dim;

    int total_bytes = packed_dim;  // = bit_width * bytes_per_plane
    for (int byte_out = tid; byte_out < total_bytes; byte_out += block_size) {
        int plane = byte_out / bytes_per_plane;
        int byte_in_plane = byte_out % bytes_per_plane;
        int dim_start = byte_in_plane * 8;

        uint8_t packed_byte = 0;
        for (int b = 0; b < 8; b++) {
            int j = dim_start + b;
            if (j >= d) break;
            int bit_val = (s_codes[j] >> plane) & 1;
            packed_byte |= (bit_val << (7 - b));
        }
        packed_row[byte_out] = packed_byte;
    }
}

// -------------------------------------------------------------------
// Host wrapper
// -------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> rotate_and_quantize_cuda(
    torch::Tensor vectors,
    torch::Tensor rotation,
    torch::Tensor boundaries,
    int64_t bit_width
) {
    c10::cuda::CUDAGuard device_guard(vectors.device());

    auto vectors_f = vectors.to(torch::kFloat32).contiguous();
    auto rotation_f = rotation.to(torch::kFloat32).contiguous();
    auto boundaries_f = boundaries.to(torch::kFloat32).contiguous();

    int N = vectors_f.size(0);
    int d = vectors_f.size(1);

    TORCH_CHECK(rotation_f.size(0) == d && rotation_f.size(1) == d,
                "Rotation matrix must be [d, d]");
    TORCH_CHECK(bit_width >= 1 && bit_width <= 4, "bit_width must be 1-4");

    int n_boundaries = boundaries_f.size(0);
    int n_levels = n_boundaries + 1;
    TORCH_CHECK(n_levels == (1 << bit_width),
                "boundaries length must be 2^bit_width - 1");

    int bytes_per_plane = (d + 7) / 8;
    int packed_dim = bit_width * bytes_per_plane;

    auto options_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(vectors.device());
    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(vectors.device());

    auto packed = torch::zeros({N, packed_dim}, options_u8);
    auto norms = torch::empty({N}, options_f32);

    if (N == 0) return {packed, norms};

    int threads = min(d, 256);
    // Shared memory: unit vector (d floats) + boundaries (n_boundaries floats) + codes (d uint8)
    int smem_bytes = (d + n_boundaries) * sizeof(float) + d * sizeof(uint8_t);

    rotate_and_quantize_kernel<<<N, threads, smem_bytes,
        at::cuda::getCurrentCUDAStream()>>>(
        vectors_f.data_ptr<float>(),
        rotation_f.data_ptr<float>(),
        boundaries_f.data_ptr<float>(),
        packed.data_ptr<uint8_t>(),
        norms.data_ptr<float>(),
        N, d, bit_width, n_boundaries, packed_dim
    );

    return {packed, norms};
}
