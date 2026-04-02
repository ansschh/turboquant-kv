/*
 * attn_scores.cu — Compute attention scores directly from packed key codes.
 *
 * Avoids materializing the full decompressed key matrix.
 * Uses precomputed centroid lookup table (fits in shared memory for b<=4).
 *
 * Input:  query [batch, heads, head_dim] float32
 *         packed_key_codes [seq_len, packed_dim] uint8
 *         key_norms [seq_len] float32
 *         centroids [2^b] float32
 *         rotation [head_dim, head_dim] float32
 *         bit_width int
 * Output: scores [batch, heads, seq_len] float32
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

// Max centroid entries for b=4 is 16
constexpr int MAX_CENTROIDS = 16;

// -------------------------------------------------------------------
// Kernel: one block per (batch, head) pair.
// Threads collaborate on the seq_len dimension.
// -------------------------------------------------------------------

__global__ void attention_scores_packed_kernel(
    const float* __restrict__ query,       // [batch * heads, head_dim]
    const uint8_t* __restrict__ packed_keys,// [seq_len, packed_dim]
    const float* __restrict__ key_norms,   // [seq_len]
    const float* __restrict__ centroids,   // [n_levels]
    const float* __restrict__ rotation,    // [head_dim, head_dim]
    float* __restrict__ scores,            // [batch * heads, seq_len]
    int batch_heads,
    int head_dim,
    int seq_len,
    int bit_width,
    int n_levels,
    int bytes_per_plane,
    int packed_dim
) {
    int bh = blockIdx.x;  // which (batch, head) pair
    if (bh >= batch_heads) return;

    extern __shared__ float smem[];
    // Layout: q_rot [head_dim] + centroids [n_levels]
    float* s_qrot = smem;
    float* s_centroids = smem + head_dim;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Load centroids into shared memory
    for (int i = tid; i < n_levels; i += block_size) {
        s_centroids[i] = centroids[i];
    }

    // Compute rotated query: q_rot = query @ rotation^T
    // i.e. q_rot[j] = sum_k query[k] * rotation^T[k,j] = sum_k query[k] * rotation[j,k]
    // rotation stored as [head_dim, head_dim] row-major, so rotation[j,k] = rotation[j*head_dim+k]
    const float* q_ptr = query + (size_t)bh * head_dim;

    for (int j = tid; j < head_dim; j += block_size) {
        float val = 0.0f;
        const float* rot_row = rotation + (size_t)j * head_dim;
        for (int k = 0; k < head_dim; k++) {
            val += q_ptr[k] * rot_row[k];
        }
        s_qrot[j] = val;
    }
    __syncthreads();

    // Now each thread handles some positions in seq_len
    float* score_ptr = scores + (size_t)bh * seq_len;

    for (int pos = tid; pos < seq_len; pos += block_size) {
        const uint8_t* packed_row = packed_keys + (size_t)pos * packed_dim;

        // Unpack codes and compute dot product with q_rot using centroid LUT
        float dot = 0.0f;

        for (int j = 0; j < head_dim; j++) {
            // Unpack code for dimension j
            int byte_idx = j / 8;
            int bit_pos = 7 - (j % 8);

            uint8_t code = 0;
            for (int i = 0; i < bit_width; i++) {
                uint8_t packed_byte = packed_row[i * bytes_per_plane + byte_idx];
                int bit_val = (packed_byte >> bit_pos) & 1;
                code |= (bit_val << i);
            }

            dot += s_qrot[j] * s_centroids[code];
        }

        // Multiply by key norm
        score_ptr[pos] = dot * key_norms[pos];
    }
}


// -------------------------------------------------------------------
// Host wrapper
// -------------------------------------------------------------------

torch::Tensor attention_scores_packed_cuda(
    torch::Tensor query,
    torch::Tensor packed_keys,
    torch::Tensor key_norms,
    torch::Tensor centroids,
    torch::Tensor rotation,
    int64_t bit_width
) {
    c10::cuda::CUDAGuard device_guard(query.device());

    // Validate inputs
    TORCH_CHECK(query.dim() == 3, "query must be [batch, heads, head_dim]");
    TORCH_CHECK(packed_keys.dim() == 2, "packed_keys must be [seq_len, packed_dim]");
    TORCH_CHECK(key_norms.dim() == 1, "key_norms must be [seq_len]");
    TORCH_CHECK(centroids.dim() == 1, "centroids must be [n_levels]");
    TORCH_CHECK(rotation.dim() == 2, "rotation must be [head_dim, head_dim]");
    TORCH_CHECK(bit_width >= 1 && bit_width <= 4, "bit_width must be 1-4");

    auto query_f = query.to(torch::kFloat32).contiguous();
    auto packed_c = packed_keys.contiguous();
    auto norms_f = key_norms.to(torch::kFloat32).contiguous();
    auto centroids_f = centroids.to(torch::kFloat32).contiguous();
    auto rotation_f = rotation.to(torch::kFloat32).contiguous();

    int batch = query_f.size(0);
    int heads = query_f.size(1);
    int head_dim = query_f.size(2);
    int seq_len = packed_c.size(0);

    TORCH_CHECK(rotation_f.size(0) == head_dim && rotation_f.size(1) == head_dim,
                "rotation must be [head_dim, head_dim]");
    TORCH_CHECK(norms_f.size(0) == seq_len, "key_norms length must match seq_len");

    int n_levels = centroids_f.size(0);
    TORCH_CHECK(n_levels == (1 << bit_width), "centroids length must be 2^bit_width");

    int bytes_per_plane = (head_dim + 7) / 8;
    int packed_dim = bit_width * bytes_per_plane;
    TORCH_CHECK(packed_c.size(1) == packed_dim,
                "packed_keys dim mismatch: expected ", packed_dim, " got ", packed_c.size(1));

    int batch_heads = batch * heads;

    auto scores = torch::empty({batch, heads, seq_len},
        torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

    if (seq_len == 0 || batch_heads == 0) return scores;

    // Flatten query to [batch*heads, head_dim]
    auto query_flat = query_f.reshape({batch_heads, head_dim});

    int threads = 256;
    int smem_bytes = (head_dim + n_levels) * sizeof(float);

    attention_scores_packed_kernel<<<batch_heads, threads, smem_bytes,
        at::cuda::getCurrentCUDAStream()>>>(
        query_flat.data_ptr<float>(),
        packed_c.data_ptr<uint8_t>(),
        norms_f.data_ptr<float>(),
        centroids_f.data_ptr<float>(),
        rotation_f.data_ptr<float>(),
        scores.data_ptr<float>(),
        batch_heads, head_dim, seq_len,
        bit_width, n_levels, bytes_per_plane, packed_dim
    );

    return scores;
}
