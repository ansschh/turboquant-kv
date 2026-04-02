/*
 * turboquant_core.h -- Standalone C++ header for TurboQuant operations.
 *
 * No PyTorch dependency.  Uses raw float/uint8_t arrays; the pybind11
 * bindings (bindings.cpp) wrap these for numpy arrays.
 *
 * Thread-safety: all const methods are safe for concurrent use.
 * Mutation (add()) is NOT thread-safe -- caller must serialise.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>

namespace turboquant {

// -----------------------------------------------------------------------
// Codebook -- Lloyd-Max for N(0, 1/sqrt(dim))
// -----------------------------------------------------------------------

struct Codebook {
    int bits;
    int dim;
    int n_levels;                   // 1 << bits
    std::vector<double> boundaries; // (n_levels - 1)  -- unscaled N(0,1) space
    std::vector<double> centroids;  // (n_levels)      -- unscaled
    double scale;                   // 1/sqrt(dim)

    // Scaled versions (float32, ready for computation)
    std::vector<float> scaled_boundaries; // (n_levels - 1)
    std::vector<float> scaled_centroids;  // (n_levels)

    Codebook() : bits(0), dim(0), n_levels(0), scale(0.0) {}
    Codebook(int bits, int dim, int max_iter = 200, double tol = 1e-12);
};

// -----------------------------------------------------------------------
// Rotation -- QR decomposition of random Gaussian matrix
// -----------------------------------------------------------------------

// Returns a row-major (dim x dim) orthogonal matrix Q.
// Uses Gram-Schmidt with re-orthogonalisation.  Deterministic via seed.
std::vector<float> make_rotation_matrix(int dim, int seed = 42);

// -----------------------------------------------------------------------
// Bit-plane pack / unpack  (matches Python reference EXACTLY)
// -----------------------------------------------------------------------

//  codes: (N x dim) uint8, values in [0, 2^bits).
//  packed: (N x packed_dim) uint8 where packed_dim = bits * ceil(dim/8).
//  Both arrays are row-major contiguous.

void pack_codes(const uint8_t* codes, int N, int dim, int bits,
                uint8_t* packed);

void unpack_codes(const uint8_t* packed, int N, int dim, int bits,
                  uint8_t* codes);

int packed_dim_for(int dim, int bits);  // = bits * ceil(dim/8)

// -----------------------------------------------------------------------
// Quantize / dequantize
// -----------------------------------------------------------------------

// quantize_vectors: normalise -> rotate -> searchsorted -> bit-pack.
//   vectors:  (N, dim) float32, row-major
//   rotation: (dim, dim) float32, row-major  (Q from make_rotation_matrix)
//   codebook: a Codebook with matching (bits, dim)
//
// Outputs (caller allocates):
//   packed: (N, packed_dim) uint8
//   norms:  (N,) float32

void quantize_vectors(const float* vectors, int N, int dim,
                      const float* rotation,
                      const Codebook& codebook,
                      uint8_t* packed, float* norms);

// dequantize_vectors: unpack -> centroid lookup -> inverse rotate -> scale.
//   packed: (N, packed_dim) uint8
//   norms:  (N,) float32
//   rotation: (dim, dim) float32, row-major
//   codebook: Codebook
//
// Output:
//   out: (N, dim) float32

void dequantize_vectors(const uint8_t* packed, const float* norms,
                        int N, int dim,
                        const float* rotation,
                        const Codebook& codebook,
                        float* out);

// -----------------------------------------------------------------------
// Search -- precomputed lookup-table approach (the fast path)
// -----------------------------------------------------------------------

// search_packed: for each query compute top-k by inner product against
//   the packed database.
//
//   queries:  (Q, dim) float32, row-major
//   packed:   (N, packed_dim) uint8 -- database packed codes
//   db_norms: (N,) float32
//   rotation: (dim, dim) float32
//   codebook: Codebook
//
// Outputs:
//   out_scores: (Q, k) float32  (descending order)
//   out_ids:    (Q, k) int64_t

void search_packed(const float* queries, int Q,
                   const uint8_t* packed, const float* db_norms, int N,
                   int dim, int bits,
                   const float* rotation,
                   const float* scaled_centroids,
                   int k,
                   float* out_scores, int64_t* out_ids);

// -----------------------------------------------------------------------
// TurboQuantIndex  -- high-level index class
// -----------------------------------------------------------------------

class TurboQuantIndex {
public:
    TurboQuantIndex(int dim, int bit_width, int seed = 42);

    // Add vectors (N, dim) float32 row-major.
    void add(const float* vectors, int N);

    // Search: returns (scores, ids) both of shape (Q, k).
    void search(const float* queries, int Q, int k,
                float* out_scores, int64_t* out_ids) const;

    // Persistence
    void save(const std::string& path) const;
    static std::unique_ptr<TurboQuantIndex> load(const std::string& path);

    // Accessors
    int dim() const { return dim_; }
    int bit_width() const { return bit_width_; }
    int seed() const { return seed_; }
    int n_vectors() const { return n_vectors_; }

    const Codebook& codebook() const { return codebook_; }
    const std::vector<float>& rotation() const { return rotation_; }
    const std::vector<uint8_t>& packed_codes() const { return packed_codes_; }
    const std::vector<float>& norms() const { return norms_; }

private:
    int dim_;
    int bit_width_;
    int seed_;
    int n_vectors_;
    int packed_dim_;

    Codebook codebook_;
    std::vector<float> rotation_;     // (dim_ * dim_)
    std::vector<uint8_t> packed_codes_; // (n_vectors_ * packed_dim_)
    std::vector<float> norms_;          // (n_vectors_)
};

}  // namespace turboquant
