/*
 * attn_values.cu — Compute attention output from packed value codes.
 *
 * Avoids materializing the full decompressed value matrix.
 *
 * Input:  attn_weights [batch, heads, seq_len] float32
 *         packed_value_codes [seq_len, packed_dim] uint8
 *         value_norms [seq_len] float32
 *         centroids [2^b] float32
 *         rotation_T [head_dim, head_dim] float32  (transpose of rotation)
 *         bit_width int
 * Output: output [batch, heads, head_dim] float32
 *
 * Algorithm:
 *   1. For each output dim j in rotated space:
 *        rotated_out[j] = sum_i attn_weights[i] * centroids[code_ij] * value_norms[i]
 *   2. output = rotated_out @ rotation_T  (inverse rotation)
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

constexpr int MAX_CENTROIDS = 16;

// -------------------------------------------------------------------
// Kernel: one block per (batch, head) pair.
// Threads collaborate on head_dim dimensions.
// -------------------------------------------------------------------

__global__ void attention_values_packed_kernel(
    const float* __restrict__ attn_weights,  // [batch_heads, seq_len]
    const uint8_t* __restrict__ packed_values,// [seq_len, packed_dim]
    const float* __restrict__ value_norms,   // [seq_len]
    const float* __restrict__ centroids,     // [n_levels]
    const float* __restrict__ rotation_T,    // [head_dim, head_dim]
    float* __restrict__ output,              // [batch_heads, head_dim]
    int batch_heads,
    int head_dim,
    int seq_len,
    int bit_width,
    int n_levels,
    int bytes_per_plane,
    int packed_dim
) {
    int bh = blockIdx.x;
    if (bh >= batch_heads) return;

    extern __shared__ float smem[];
    // Layout: centroids [n_levels] + rotated_out [head_dim]
    float* s_centroids = smem;
    float* s_rotated = smem + n_levels;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Load centroids
    for (int i = tid; i < n_levels; i += block_size) {
        s_centroids[i] = centroids[i];
    }

    // Zero rotated output
    for (int j = tid; j < head_dim; j += block_size) {
        s_rotated[j] = 0.0f;
    }
    __syncthreads();

    const float* w_ptr = attn_weights + (size_t)bh * seq_len;

    // Each thread processes a subset of output dimensions for ALL seq positions.
    // This ensures good memory access for packed_values.
    for (int j = tid; j < head_dim; j += block_size) {
        float accum = 0.0f;

        int byte_idx = j / 8;
        int bit_pos = 7 - (j % 8);

        for (int pos = 0; pos < seq_len; pos++) {
            const uint8_t* packed_row = packed_values + (size_t)pos * packed_dim;

            // Unpack code for dimension j of value at position pos
            uint8_t code = 0;
            for (int i = 0; i < bit_width; i++) {
                uint8_t packed_byte = packed_row[i * bytes_per_plane + byte_idx];
                int bit_val = (packed_byte >> bit_pos) & 1;
                code |= (bit_val << i);
            }

            accum += w_ptr[pos] * s_centroids[code] * value_norms[pos];
        }

        s_rotated[j] = accum;
    }
    __syncthreads();

    // Apply inverse rotation: output[k] = sum_j s_rotated[j] * rotation_T[j][k]
    // rotation_T is [head_dim, head_dim], row-major
    // output[k] = sum_j s_rotated[j] * rotation_T[j*head_dim + k]
    // Equivalently: output = s_rotated @ rotation_T
    // So output[k] = sum_j s_rotated[j] * rotation_T[j][k]

    float* out_ptr = output + (size_t)bh * head_dim;

    for (int k = tid; k < head_dim; k += block_size) {
        float val = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            val += s_rotated[j] * rotation_T[j * head_dim + k];
        }
        out_ptr[k] = val;
    }
}


// -------------------------------------------------------------------
// Host wrapper
// -------------------------------------------------------------------

torch::Tensor attention_values_packed_cuda(
    torch::Tensor attn_weights,
    torch::Tensor packed_values,
    torch::Tensor value_norms,
    torch::Tensor centroids,
    torch::Tensor rotation_T,
    int64_t bit_width
) {
    c10::cuda::CUDAGuard device_guard(attn_weights.device());

    TORCH_CHECK(attn_weights.dim() == 3, "attn_weights must be [batch, heads, seq_len]");
    TORCH_CHECK(packed_values.dim() == 2, "packed_values must be [seq_len, packed_dim]");
    TORCH_CHECK(value_norms.dim() == 1, "value_norms must be [seq_len]");
    TORCH_CHECK(centroids.dim() == 1, "centroids must be [n_levels]");
    TORCH_CHECK(rotation_T.dim() == 2, "rotation_T must be [head_dim, head_dim]");
    TORCH_CHECK(bit_width >= 1 && bit_width <= 4, "bit_width must be 1-4");

    auto weights_f = attn_weights.to(torch::kFloat32).contiguous();
    auto packed_c = packed_values.contiguous();
    auto norms_f = value_norms.to(torch::kFloat32).contiguous();
    auto centroids_f = centroids.to(torch::kFloat32).contiguous();
    auto rotation_T_f = rotation_T.to(torch::kFloat32).contiguous();

    int batch = weights_f.size(0);
    int heads = weights_f.size(1);
    int seq_len = weights_f.size(2);
    int head_dim = rotation_T_f.size(0);

    TORCH_CHECK(rotation_T_f.size(1) == head_dim, "rotation_T must be square");
    TORCH_CHECK(norms_f.size(0) == seq_len, "value_norms length must match seq_len");
    TORCH_CHECK(packed_c.size(0) == seq_len, "packed_values rows must match seq_len");

    int n_levels = centroids_f.size(0);
    TORCH_CHECK(n_levels == (1 << bit_width), "centroids length must be 2^bit_width");

    int bytes_per_plane = (head_dim + 7) / 8;
    int packed_dim = bit_width * bytes_per_plane;
    TORCH_CHECK(packed_c.size(1) == packed_dim,
                "packed_values dim mismatch: expected ", packed_dim, " got ", packed_c.size(1));

    int batch_heads = batch * heads;

    auto output = torch::empty({batch, heads, head_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(attn_weights.device()));

    if (seq_len == 0 || batch_heads == 0) return output;

    // Flatten weights to [batch*heads, seq_len]
    auto weights_flat = weights_f.reshape({batch_heads, seq_len});

    int threads = min(head_dim, 256);
    int smem_bytes = (n_levels + head_dim) * sizeof(float);

    attention_values_packed_kernel<<<batch_heads, threads, smem_bytes,
        at::cuda::getCurrentCUDAStream()>>>(
        weights_flat.data_ptr<float>(),
        packed_c.data_ptr<uint8_t>(),
        norms_f.data_ptr<float>(),
        centroids_f.data_ptr<float>(),
        rotation_T_f.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_heads, head_dim, seq_len,
        bit_width, n_levels, bytes_per_plane, packed_dim
    );

    return output;
}
