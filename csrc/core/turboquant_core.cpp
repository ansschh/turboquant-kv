/*
 * turboquant_core.cpp -- Standalone C++ implementation of TurboQuant.
 *
 * No PyTorch, no Eigen.  Uses OpenMP for parallelism.
 * LAPACK is optional (for QR); a pure-C++ Gram-Schmidt fallback is provided.
 */

#include "turboquant_core.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace turboquant {

// =====================================================================
//  Helpers
// =====================================================================

static constexpr double PI = 3.14159265358979323846;

// Standard-normal PDF / CDF (double precision, good enough for codebook).
static double phi_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * PI);
}

// Abramowitz & Stegun approximation -- max error ~1.5e-7.
static double phi_cdf(double x) {
    if (x < -8.0) return 0.0;
    if (x >  8.0) return 1.0;

    // Use the symmetry phi(-x) = 1 - phi(x)
    bool neg = (x < 0.0);
    double z = neg ? -x : x;

    const double p  = 0.2316419;
    const double b1 = 0.319381530;
    const double b2 = -0.356563782;
    const double b3 = 1.781477937;
    const double b4 = -1.821255978;
    const double b5 = 1.330274429;

    double t = 1.0 / (1.0 + p * z);
    double t2 = t * t;
    double t3 = t2 * t;
    double t4 = t3 * t;
    double t5 = t4 * t;
    double poly = b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5;
    double cdf = 1.0 - phi_pdf(z) * poly;

    return neg ? (1.0 - cdf) : cdf;
}

// =====================================================================
//  Codebook -- Lloyd-Max for N(0, 1) then scale by 1/sqrt(dim)
// =====================================================================

Codebook::Codebook(int bits_, int dim_, int max_iter, double tol)
    : bits(bits_), dim(dim_), n_levels(1 << bits_), scale(1.0 / std::sqrt(static_cast<double>(dim_)))
{
    // Initial centroids: uniform in [-3, 3]
    centroids.resize(n_levels);
    for (int i = 0; i < n_levels; i++) {
        centroids[i] = -3.0 + 6.0 * i / (n_levels - 1);
    }

    boundaries.resize(n_levels - 1);

    for (int iter = 0; iter < max_iter; iter++) {
        // Boundaries = midpoints of consecutive centroids
        for (int i = 0; i < n_levels - 1; i++) {
            boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
        }

        // Build edge array: [-inf, b0, b1, ..., b_{L-2}, +inf]
        std::vector<double> edges(n_levels + 1);
        edges[0] = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < n_levels - 1; i++) {
            edges[i + 1] = boundaries[i];
        }
        edges[n_levels] = std::numeric_limits<double>::infinity();

        // Update centroids: E[X | lo < X < hi]
        std::vector<double> new_centroids(n_levels);
        double max_diff = 0.0;

        for (int i = 0; i < n_levels; i++) {
            double lo = edges[i];
            double hi = edges[i + 1];
            double prob = phi_cdf(hi) - phi_cdf(lo);
            if (prob < 1e-15) {
                new_centroids[i] = centroids[i];
            } else {
                // E[X | lo < X < hi] = (phi(lo) - phi(hi)) / prob
                new_centroids[i] = (phi_pdf(lo) - phi_pdf(hi)) / prob;
            }
            max_diff = std::max(max_diff, std::abs(new_centroids[i] - centroids[i]));
        }

        centroids = new_centroids;
        if (max_diff < tol) break;
    }

    // Final boundaries
    for (int i = 0; i < n_levels - 1; i++) {
        boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
    }

    // Scaled versions
    scaled_boundaries.resize(n_levels - 1);
    scaled_centroids.resize(n_levels);
    for (int i = 0; i < n_levels - 1; i++) {
        scaled_boundaries[i] = static_cast<float>(boundaries[i] * scale);
    }
    for (int i = 0; i < n_levels; i++) {
        scaled_centroids[i] = static_cast<float>(centroids[i] * scale);
    }
}

// =====================================================================
//  Rotation matrix -- QR via modified Gram-Schmidt
// =====================================================================

std::vector<float> make_rotation_matrix(int dim, int seed) {
    // Generate random Gaussian matrix using the SAME RNG as numpy.RandomState.
    // numpy.RandomState uses MT19937 with the same seed convention, but
    // the Box-Muller pairing and state mapping differ.  To guarantee
    // determinism we replicate numpy's approach: use MT19937 + Box-Muller
    // with the SAME ordering numpy uses.  In practice the exact values
    // don't matter as long as they're consistent within the C++ world;
    // the Python side will use its own rotation.  Determinism within C++
    // is guaranteed by std::mt19937 + seed.

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::normal_distribution<float> normal(0.0f, 1.0f);

    const int n = dim;
    std::vector<float> G(n * n);
    for (int i = 0; i < n * n; i++) {
        G[i] = normal(rng);
    }

    // Modified Gram-Schmidt with re-orthogonalisation (2 passes)
    // Q stored column-major during construction, then transposed to row-major.
    // Actually easier: work directly on columns of G (row-major).
    // Column j of G (row-major) = G[row*n + j] for row = 0..n-1.
    // We orthonormalise columns in place.

    // Helper lambdas operating on column slices of G (row-major, stride = n).
    auto col_dot = [&](int c1, int c2) -> double {
        double s = 0.0;
        for (int r = 0; r < n; r++) s += (double)G[r * n + c1] * (double)G[r * n + c2];
        return s;
    };
    auto col_norm = [&](int c) -> double { return std::sqrt(col_dot(c, c)); };
    auto col_sub = [&](int target, int source, double coeff) {
        for (int r = 0; r < n; r++) G[r * n + target] -= (float)(coeff * G[r * n + source]);
    };
    auto col_scale = [&](int c, double s) {
        for (int r = 0; r < n; r++) G[r * n + c] = (float)((double)G[r * n + c] * s);
    };

    for (int j = 0; j < n; j++) {
        // Two passes of MGS for numerical stability
        for (int pass = 0; pass < 2; pass++) {
            for (int i = 0; i < j; i++) {
                double d = col_dot(j, i);
                col_sub(j, i, d);
            }
        }
        double nrm = col_norm(j);
        if (nrm < 1e-12) {
            // Degenerate -- shouldn't happen with random Gaussian
            throw std::runtime_error("make_rotation_matrix: degenerate column");
        }
        col_scale(j, 1.0 / nrm);
    }

    // G is now orthogonal (columns are orthonormal).
    // Enforce positive diagonal convention (matching numpy QR sign convention):
    // For each column j, if G[j][j] < 0 then flip the entire column.
    for (int j = 0; j < n; j++) {
        if (G[j * n + j] < 0.0f) {
            for (int r = 0; r < n; r++) G[r * n + j] = -G[r * n + j];
        }
    }

    // G is now (n x n) row-major where columns are orthonormal.
    // This is equivalent to Q from QR.  The reference stores Q as (dim, dim)
    // row-major where Q[i] is the i-th row.  Our G already has that layout
    // because column-orthonormal in row-major == row-orthonormal when
    // transposed... but actually we want Q such that Q @ Q^T = I with rows
    // being the basis.  Let's just verify: G has orthonormal columns, so
    // G^T G = I.  The reference uses Q from np.linalg.qr which satisfies
    // Q^T Q = I (orthonormal columns).  The rotation is applied as
    // rotated = unit @ Q^T.  So we return G which has orthonormal columns.
    // When the user computes x @ G^T, they get the rotation.

    return G;
}

// =====================================================================
//  Bit-plane packing / unpacking
// =====================================================================

int packed_dim_for(int dim, int bits) {
    return bits * ((dim + 7) / 8);
}

void pack_codes(const uint8_t* codes, int N, int dim, int bits,
                uint8_t* packed) {
    const int bytes_per_plane = (dim + 7) / 8;
    const int pdim = bits * bytes_per_plane;

    // Zero the output
    std::memset(packed, 0, (size_t)N * pdim);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int n = 0; n < N; n++) {
        const uint8_t* codes_row = codes + (size_t)n * dim;
        uint8_t* packed_row = packed + (size_t)n * pdim;

        for (int j = 0; j < dim; j++) {
            uint8_t code = codes_row[j];
            int byte_idx = j / 8;
            int bit_pos = 7 - (j % 8);

            for (int i = 0; i < bits; i++) {
                if ((code >> i) & 1) {
                    packed_row[i * bytes_per_plane + byte_idx] |= (1 << bit_pos);
                }
            }
        }
    }
}

void unpack_codes(const uint8_t* packed, int N, int dim, int bits,
                  uint8_t* codes) {
    const int bytes_per_plane = (dim + 7) / 8;
    const int pdim = bits * bytes_per_plane;

    #pragma omp parallel for schedule(dynamic, 64)
    for (int n = 0; n < N; n++) {
        const uint8_t* packed_row = packed + (size_t)n * pdim;
        uint8_t* codes_row = codes + (size_t)n * dim;

        for (int j = 0; j < dim; j++) {
            int byte_idx = j / 8;
            int bit_pos = 7 - (j % 8);
            uint8_t code = 0;
            for (int i = 0; i < bits; i++) {
                uint8_t byte_val = packed_row[i * bytes_per_plane + byte_idx];
                code |= (((byte_val >> bit_pos) & 1) << i);
            }
            codes_row[j] = code;
        }
    }
}

// =====================================================================
//  Quantize / dequantize
// =====================================================================

void quantize_vectors(const float* vectors, int N, int dim,
                      const float* rotation,
                      const Codebook& codebook,
                      uint8_t* packed, float* norms) {
    const int bits = codebook.bits;
    const int n_boundaries = codebook.n_levels - 1;
    const float* bnd = codebook.scaled_boundaries.data();
    const int bytes_per_plane = (dim + 7) / 8;
    const int pdim = bits * bytes_per_plane;

    // Zero packed output
    std::memset(packed, 0, (size_t)N * pdim);

    #pragma omp parallel for schedule(dynamic, 32)
    for (int n = 0; n < N; n++) {
        const float* v = vectors + (size_t)n * dim;

        // 1. Compute norm
        float norm_sq = 0.0f;
        for (int k = 0; k < dim; k++) norm_sq += v[k] * v[k];
        float norm_val = std::sqrt(norm_sq);
        norms[n] = norm_val;
        float inv_norm = (norm_val > 1e-10f) ? 1.0f / norm_val : 0.0f;

        uint8_t* packed_row = packed + (size_t)n * pdim;

        // 2. For each rotated dimension j
        for (int j = 0; j < dim; j++) {
            // rotated[j] = sum_k unit[k] * rotation[j * dim + k]
            // rotation is stored row-major: Q[j][k]
            // We want rotated = unit @ Q^T, so rotated[j] = sum_k unit[k] * Q[j][k]
            // Wait -- the reference does: rotated = unit @ rotation.t()
            // rotation is (dim, dim), rotation.t() is transpose.
            // rotated[j] = sum_k unit[k] * rotation^T[k][j] = sum_k unit[k] * rotation[j][k]
            // So rotated[j] = dot(unit, rotation_row_j).
            float val = 0.0f;
            const float* rot_row = rotation + (size_t)j * dim;
            for (int k = 0; k < dim; k++) {
                val += (v[k] * inv_norm) * rot_row[k];
            }

            // 3. searchsorted on sorted boundaries (ascending)
            //    Find first boundary > val, code = that index
            int code = 0;
            // Binary search for upper_bound
            int lo = 0, hi = n_boundaries;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (bnd[mid] <= val) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            code = lo;  // number of boundaries <= val

            // 4. Bit-pack
            int byte_idx = j / 8;
            int bit_pos = 7 - (j % 8);
            for (int i = 0; i < bits; i++) {
                if ((code >> i) & 1) {
                    packed_row[i * bytes_per_plane + byte_idx] |= (uint8_t)(1 << bit_pos);
                }
            }
        }
    }
}

void dequantize_vectors(const uint8_t* packed, const float* norms,
                        int N, int dim,
                        const float* rotation,
                        const Codebook& codebook,
                        float* out) {
    const int bits = codebook.bits;
    const float* cents = codebook.scaled_centroids.data();
    const int bytes_per_plane = (dim + 7) / 8;
    const int pdim = bits * bytes_per_plane;

    #pragma omp parallel for schedule(dynamic, 32)
    for (int n = 0; n < N; n++) {
        const uint8_t* packed_row = packed + (size_t)n * pdim;
        float* out_row = out + (size_t)n * dim;
        float norm_val = norms[n];

        // 1. Unpack codes and lookup centroids to get rotated reconstruction
        //    Then inverse rotate: reconstructed[k] = sum_j centroid_val_j * rotation[j][k]
        //    (Because inverse of orthogonal Q is Q^T, and the forward rotation was
        //     rotated = unit @ Q^T, so unit = rotated @ Q, meaning
        //     unit[k] = sum_j rotated[j] * Q[j][k])

        // Accumulate directly: out[k] = norm * sum_j cents[code_j] * rotation[j][k]
        // = norm * sum_j cents[code_j] * Q[j][k]
        // This is out = norm * (rotated_recon @ Q)

        // First pass: compute centroid values for each dimension
        // We can fuse by iterating over k and accumulating.
        // But that requires random access to codes for each j inside k loop.
        // Better: unpack codes first, then matmul.

        // Unpack codes into a small buffer
        // For typical dims (128-256), this fits in stack/L1.
        // Use alloca-like approach or just a vector. Since we're in OMP,
        // let's use a thread-local stack buffer (max dim ~2048 should be fine).
        uint8_t codes_buf[4096];  // max dim supported
        for (int j = 0; j < dim; j++) {
            int byte_idx = j / 8;
            int bit_pos = 7 - (j % 8);
            uint8_t code = 0;
            for (int i = 0; i < bits; i++) {
                uint8_t byte_val = packed_row[i * bytes_per_plane + byte_idx];
                code |= (((byte_val >> bit_pos) & 1) << i);
            }
            codes_buf[j] = code;
        }

        // Now compute: out[k] = norm * sum_j cents[codes_buf[j]] * rotation[j * dim + k]
        for (int k = 0; k < dim; k++) {
            float val = 0.0f;
            for (int j = 0; j < dim; j++) {
                val += cents[codes_buf[j]] * rotation[j * dim + k];
            }
            out_row[k] = norm_val * val;
        }
    }
}

// =====================================================================
//  Search -- precomputed lookup table
// =====================================================================

void search_packed(const float* queries, int Q,
                   const uint8_t* packed, const float* db_norms, int N,
                   int dim, int bits,
                   const float* rotation,
                   const float* scaled_centroids,
                   int k,
                   float* out_scores, int64_t* out_ids) {
    const int n_levels = 1 << bits;
    const int bytes_per_plane = (dim + 7) / 8;
    const int pdim = bits * bytes_per_plane;
    const int actual_k = std::min(k, N);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int q = 0; q < Q; q++) {
        const float* query = queries + (size_t)q * dim;

        // 1. Rotate query: q_rot[j] = sum_k query[k] * rotation[j][k]
        //    (same as query @ rotation^T, where rotation[j] is row j)
        std::vector<float> q_rot(dim);
        for (int j = 0; j < dim; j++) {
            float val = 0.0f;
            const float* rot_row = rotation + (size_t)j * dim;
            for (int kk = 0; kk < dim; kk++) {
                val += query[kk] * rot_row[kk];
            }
            q_rot[j] = val;
        }

        // 2. Build lookup table: table[j * n_levels + c] = q_rot[j] * centroids[c]
        std::vector<float> table(dim * n_levels);
        for (int j = 0; j < dim; j++) {
            float qj = q_rot[j];
            for (int c = 0; c < n_levels; c++) {
                table[j * n_levels + c] = qj * scaled_centroids[c];
            }
        }

        // 3. Score each database vector using lookup table
        //    score_i = db_norms[i] * sum_j table[j][code_i_j]
        //    Maintain top-k via a min-heap.

        // Min-heap: (score, index)
        using ScoreIdx = std::pair<float, int64_t>;
        auto cmp = [](const ScoreIdx& a, const ScoreIdx& b) {
            return a.first > b.first;  // min-heap
        };
        std::priority_queue<ScoreIdx, std::vector<ScoreIdx>, decltype(cmp)> heap(cmp);

        for (int i = 0; i < N; i++) {
            const uint8_t* packed_row = packed + (size_t)i * pdim;

            // Accumulate score via byte-level unpacking
            float dot = 0.0f;

            // Process byte by byte across all bit planes simultaneously
            for (int byte_idx = 0; byte_idx < bytes_per_plane; byte_idx++) {
                // Load bit-plane bytes
                uint8_t plane_bytes[8]; // max 8 bits
                for (int p = 0; p < bits; p++) {
                    plane_bytes[p] = packed_row[p * bytes_per_plane + byte_idx];
                }

                // Process 8 dimensions per byte
                int base_dim = byte_idx * 8;
                int dims_in_byte = std::min(8, dim - base_dim);

                for (int b = 0; b < dims_in_byte; b++) {
                    int bit_pos = 7 - b;
                    int d_idx = base_dim + b;

                    // Extract code from bit planes
                    uint8_t code = 0;
                    for (int p = 0; p < bits; p++) {
                        code |= (((plane_bytes[p] >> bit_pos) & 1) << p);
                    }

                    dot += table[d_idx * n_levels + code];
                }
            }

            float score = dot * db_norms[i];

            if ((int)heap.size() < actual_k) {
                heap.push({score, (int64_t)i});
            } else if (score > heap.top().first) {
                heap.pop();
                heap.push({score, (int64_t)i});
            }
        }

        // Extract results in descending order
        float* q_scores = out_scores + (size_t)q * k;
        int64_t* q_ids = out_ids + (size_t)q * k;

        int n_results = (int)heap.size();
        // Fill from the end (heap gives ascending order when popped)
        for (int i = n_results - 1; i >= 0; i--) {
            q_scores[i] = heap.top().first;
            q_ids[i] = heap.top().second;
            heap.pop();
        }
        // Fill remaining slots with -inf / -1
        for (int i = n_results; i < k; i++) {
            q_scores[i] = -std::numeric_limits<float>::infinity();
            q_ids[i] = -1;
        }
    }
}

// =====================================================================
//  TurboQuantIndex
// =====================================================================

static const uint32_t TQIDX_MAGIC = 0x54514958;  // "TQIX"
static const uint32_t TQIDX_VERSION = 1;

TurboQuantIndex::TurboQuantIndex(int dim, int bit_width, int seed)
    : dim_(dim), bit_width_(bit_width), seed_(seed), n_vectors_(0),
      packed_dim_(packed_dim_for(dim, bit_width)),
      codebook_(bit_width, dim),
      rotation_(make_rotation_matrix(dim, seed))
{}

void TurboQuantIndex::add(const float* vectors, int N) {
    size_t old_size = packed_codes_.size();
    size_t new_packed_bytes = (size_t)N * packed_dim_;
    size_t old_n = n_vectors_;

    packed_codes_.resize(old_size + new_packed_bytes);
    norms_.resize(old_n + N);

    quantize_vectors(vectors, N, dim_,
                     rotation_.data(), codebook_,
                     packed_codes_.data() + old_size,
                     norms_.data() + old_n);

    n_vectors_ += N;
}

void TurboQuantIndex::search(const float* queries, int Q, int k,
                             float* out_scores, int64_t* out_ids) const {
    if (n_vectors_ == 0) {
        throw std::runtime_error("Index is empty");
    }
    search_packed(queries, Q,
                  packed_codes_.data(), norms_.data(), n_vectors_,
                  dim_, bit_width_,
                  rotation_.data(),
                  codebook_.scaled_centroids.data(),
                  k, out_scores, out_ids);
}

void TurboQuantIndex::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for writing: " + path);

    // Header: magic(4) + version(4) + dim(4) + bit_width(4) + seed(4) + n_vectors(4) = 24 bytes
    auto write_u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    write_u32(TQIDX_MAGIC);
    write_u32(TQIDX_VERSION);
    write_u32(static_cast<uint32_t>(dim_));
    write_u32(static_cast<uint32_t>(bit_width_));
    write_u32(static_cast<uint32_t>(seed_));
    write_u32(static_cast<uint32_t>(n_vectors_));

    // Packed codes
    f.write(reinterpret_cast<const char*>(packed_codes_.data()), packed_codes_.size());
    // Norms
    f.write(reinterpret_cast<const char*>(norms_.data()), norms_.size() * sizeof(float));

    if (!f) throw std::runtime_error("Write error: " + path);
}

std::unique_ptr<TurboQuantIndex> TurboQuantIndex::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for reading: " + path);

    auto read_u32 = [&]() -> uint32_t {
        uint32_t v;
        f.read(reinterpret_cast<char*>(&v), 4);
        return v;
    };

    uint32_t magic = read_u32();
    if (magic != TQIDX_MAGIC) {
        throw std::runtime_error("Invalid TQIDX file (bad magic)");
    }
    uint32_t version = read_u32();
    if (version != TQIDX_VERSION) {
        throw std::runtime_error("Unsupported TQIDX version");
    }

    int dim = static_cast<int>(read_u32());
    int bit_width = static_cast<int>(read_u32());
    int seed = static_cast<int>(read_u32());
    int n_vectors = static_cast<int>(read_u32());

    auto idx = std::make_unique<TurboQuantIndex>(dim, bit_width, seed);
    idx->n_vectors_ = n_vectors;

    int pdim = packed_dim_for(dim, bit_width);
    size_t packed_bytes = (size_t)n_vectors * pdim;
    idx->packed_codes_.resize(packed_bytes);
    f.read(reinterpret_cast<char*>(idx->packed_codes_.data()), packed_bytes);

    idx->norms_.resize(n_vectors);
    f.read(reinterpret_cast<char*>(idx->norms_.data()), n_vectors * sizeof(float));

    if (!f) throw std::runtime_error("Read error / truncated file: " + path);

    return idx;
}

}  // namespace turboquant
