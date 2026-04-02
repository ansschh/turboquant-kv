/*
 * quant_cpu.cpp — CPU fallback implementations using OpenMP.
 *
 * Provides:
 *   rotate_and_quantize_cpu
 *   attention_scores_packed_cpu
 *   attention_values_packed_cpu
 *   pack_codes_cpu
 *   unpack_codes_cpu
 */

#include <torch/extension.h>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

// -------------------------------------------------------------------
// Helper: unpack a single code at position j from a packed row
// -------------------------------------------------------------------

static inline uint8_t unpack_single_code(
    const uint8_t* packed_row,
    int j, int bit_width, int bytes_per_plane
) {
    int byte_idx = j / 8;
    int bit_pos = 7 - (j % 8);
    uint8_t code = 0;
    for (int i = 0; i < bit_width; i++) {
        uint8_t packed_byte = packed_row[i * bytes_per_plane + byte_idx];
        int bit_val = (packed_byte >> bit_pos) & 1;
        code |= (bit_val << i);
    }
    return code;
}


// -------------------------------------------------------------------
// rotate_and_quantize_cpu
// -------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> rotate_and_quantize_cpu(
    torch::Tensor vectors,
    torch::Tensor rotation,
    torch::Tensor boundaries,
    int64_t bit_width
) {
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

    auto packed = torch::zeros({N, packed_dim}, torch::kUInt8);
    auto norms = torch::empty({N}, torch::kFloat32);

    float* vec_ptr = vectors_f.data_ptr<float>();
    float* rot_ptr = rotation_f.data_ptr<float>();
    float* bnd_ptr = boundaries_f.data_ptr<float>();
    uint8_t* pack_ptr = packed.data_ptr<uint8_t>();
    float* norm_ptr = norms.data_ptr<float>();

    #pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < N; n++) {
        float* v = vec_ptr + (size_t)n * d;

        // Compute norm
        float norm_sq = 0.0f;
        for (int k = 0; k < d; k++) norm_sq += v[k] * v[k];
        float norm_val = sqrtf(norm_sq);
        norm_ptr[n] = norm_val;
        float inv_norm = (norm_val > 1e-10f) ? 1.0f / norm_val : 0.0f;

        uint8_t* packed_row = pack_ptr + (size_t)n * packed_dim;

        for (int j = 0; j < d; j++) {
            // Compute rotated[j] = sum_k unit[k] * rotation[j][k]
            float val = 0.0f;
            for (int k = 0; k < d; k++) {
                val += (v[k] * inv_norm) * rot_ptr[j * d + k];
            }

            // Searchsorted
            int code = 0;
            for (int b = 0; b < n_boundaries; b++) {
                if (val > bnd_ptr[b]) code = b + 1;
                else break;
            }

            // Bit-pack
            int byte_idx = j / 8;
            int bit_pos = 7 - (j % 8);
            for (int i = 0; i < bit_width; i++) {
                int bit_val = (code >> i) & 1;
                if (bit_val) {
                    packed_row[i * bytes_per_plane + byte_idx] |= (1 << bit_pos);
                }
            }
        }
    }

    return {packed, norms};
}


// -------------------------------------------------------------------
// attention_scores_packed_cpu
// -------------------------------------------------------------------

torch::Tensor attention_scores_packed_cpu(
    torch::Tensor query,
    torch::Tensor packed_keys,
    torch::Tensor key_norms,
    torch::Tensor centroids,
    torch::Tensor rotation,
    int64_t bit_width
) {
    TORCH_CHECK(query.dim() == 3, "query must be [batch, heads, head_dim]");
    TORCH_CHECK(packed_keys.dim() == 2, "packed_keys must be [seq_len, packed_dim]");
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
    int n_levels = centroids_f.size(0);

    int bytes_per_plane = (head_dim + 7) / 8;
    int packed_dim = bit_width * bytes_per_plane;

    auto scores = torch::empty({batch, heads, seq_len}, torch::kFloat32);

    float* q_ptr = query_f.data_ptr<float>();
    uint8_t* pk_ptr = packed_c.data_ptr<uint8_t>();
    float* kn_ptr = norms_f.data_ptr<float>();
    float* cen_ptr = centroids_f.data_ptr<float>();
    float* rot_ptr = rotation_f.data_ptr<float>();
    float* sc_ptr = scores.data_ptr<float>();

    int batch_heads = batch * heads;

    #pragma omp parallel for schedule(dynamic)
    for (int bh = 0; bh < batch_heads; bh++) {
        // Rotate query
        std::vector<float> q_rot(head_dim);
        float* q_bh = q_ptr + (size_t)bh * head_dim;
        for (int j = 0; j < head_dim; j++) {
            float val = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                val += rot_ptr[j * head_dim + k] * q_bh[k];
            }
            q_rot[j] = val;
        }

        // Score each position
        float* sc_bh = sc_ptr + (size_t)bh * seq_len;
        for (int pos = 0; pos < seq_len; pos++) {
            const uint8_t* packed_row = pk_ptr + (size_t)pos * packed_dim;
            float dot = 0.0f;

            for (int j = 0; j < head_dim; j++) {
                uint8_t code = unpack_single_code(packed_row, j, bit_width, bytes_per_plane);
                dot += q_rot[j] * cen_ptr[code];
            }

            sc_bh[pos] = dot * kn_ptr[pos];
        }
    }

    return scores;
}


// -------------------------------------------------------------------
// attention_values_packed_cpu
// -------------------------------------------------------------------

torch::Tensor attention_values_packed_cpu(
    torch::Tensor attn_weights,
    torch::Tensor packed_values,
    torch::Tensor value_norms,
    torch::Tensor centroids,
    torch::Tensor rotation_T,
    int64_t bit_width
) {
    TORCH_CHECK(attn_weights.dim() == 3, "attn_weights must be [batch, heads, seq_len]");
    TORCH_CHECK(packed_values.dim() == 2, "packed_values must be [seq_len, packed_dim]");
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
    int n_levels = centroids_f.size(0);

    int bytes_per_plane = (head_dim + 7) / 8;
    int packed_dim = bit_width * bytes_per_plane;

    auto output = torch::empty({batch, heads, head_dim}, torch::kFloat32);

    float* w_ptr = weights_f.data_ptr<float>();
    uint8_t* pv_ptr = packed_c.data_ptr<uint8_t>();
    float* vn_ptr = norms_f.data_ptr<float>();
    float* cen_ptr = centroids_f.data_ptr<float>();
    float* rotT_ptr = rotation_T_f.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    int batch_heads = batch * heads;

    #pragma omp parallel for schedule(dynamic)
    for (int bh = 0; bh < batch_heads; bh++) {
        float* w_bh = w_ptr + (size_t)bh * seq_len;

        // Step 1: accumulate in rotated space
        std::vector<float> rotated_out(head_dim, 0.0f);

        for (int j = 0; j < head_dim; j++) {
            float accum = 0.0f;
            for (int pos = 0; pos < seq_len; pos++) {
                const uint8_t* packed_row = pv_ptr + (size_t)pos * packed_dim;
                uint8_t code = unpack_single_code(packed_row, j, bit_width, bytes_per_plane);
                accum += w_bh[pos] * cen_ptr[code] * vn_ptr[pos];
            }
            rotated_out[j] = accum;
        }

        // Step 2: inverse rotation
        float* o_bh = out_ptr + (size_t)bh * head_dim;
        for (int k = 0; k < head_dim; k++) {
            float val = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                val += rotated_out[j] * rotT_ptr[j * head_dim + k];
            }
            o_bh[k] = val;
        }
    }

    return output;
}


// -------------------------------------------------------------------
// pack_codes_cpu
// -------------------------------------------------------------------

torch::Tensor pack_codes_cpu(torch::Tensor codes, int64_t bit_width) {
    TORCH_CHECK(codes.dtype() == torch::kUInt8, "codes must be uint8");
    TORCH_CHECK(codes.dim() == 2, "codes must be 2D [N, d]");
    TORCH_CHECK(bit_width >= 1 && bit_width <= 4, "bit_width must be 1-4");

    auto codes_c = codes.contiguous();
    int N = codes_c.size(0);
    int d = codes_c.size(1);
    int bytes_per_plane = (d + 7) / 8;
    int packed_dim = bit_width * bytes_per_plane;

    auto packed = torch::zeros({N, packed_dim}, torch::kUInt8);

    uint8_t* c_ptr = codes_c.data_ptr<uint8_t>();
    uint8_t* p_ptr = packed.data_ptr<uint8_t>();

    #pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < N; n++) {
        uint8_t* codes_row = c_ptr + (size_t)n * d;
        uint8_t* packed_row = p_ptr + (size_t)n * packed_dim;

        for (int j = 0; j < d; j++) {
            uint8_t code = codes_row[j];
            int byte_idx = j / 8;
            int bit_pos = 7 - (j % 8);

            for (int i = 0; i < bit_width; i++) {
                int bit_val = (code >> i) & 1;
                if (bit_val) {
                    packed_row[i * bytes_per_plane + byte_idx] |= (1 << bit_pos);
                }
            }
        }
    }

    return packed;
}


// -------------------------------------------------------------------
// unpack_codes_cpu
// -------------------------------------------------------------------

torch::Tensor unpack_codes_cpu(torch::Tensor packed, int64_t bit_width, int64_t dim) {
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(packed.dim() == 2, "packed must be 2D");
    TORCH_CHECK(bit_width >= 1 && bit_width <= 4, "bit_width must be 1-4");

    auto packed_c = packed.contiguous();
    int N = packed_c.size(0);
    int bytes_per_plane = (dim + 7) / 8;
    int packed_dim = bit_width * bytes_per_plane;

    TORCH_CHECK(packed_c.size(1) == packed_dim,
                "packed dim mismatch: expected ", packed_dim, " got ", packed_c.size(1));

    auto codes = torch::zeros({N, dim}, torch::kUInt8);

    uint8_t* p_ptr = packed_c.data_ptr<uint8_t>();
    uint8_t* c_ptr = codes.data_ptr<uint8_t>();

    #pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < N; n++) {
        const uint8_t* packed_row = p_ptr + (size_t)n * packed_dim;
        uint8_t* codes_row = c_ptr + (size_t)n * dim;

        for (int j = 0; j < (int)dim; j++) {
            codes_row[j] = unpack_single_code(packed_row, j, bit_width, bytes_per_plane);
        }
    }

    return codes;
}
