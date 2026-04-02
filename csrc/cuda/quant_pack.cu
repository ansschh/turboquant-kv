/*
 * quant_pack.cu — Standalone quantize+pack and unpack kernels.
 *
 * pack_codes_cuda:   (N, d) uint8 codes + bit_width -> (N, packed_dim) uint8
 * unpack_codes_cuda: (N, packed_dim) uint8 + bit_width + dim -> (N, dim) uint8
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

// -------------------------------------------------------------------
// Pack kernel: each thread handles one output byte
// This avoids atomics entirely — each output byte is written by exactly one thread.
// -------------------------------------------------------------------

__global__ void pack_codes_kernel(
    const uint8_t* __restrict__ codes,  // [N, d]
    uint8_t* __restrict__ packed,       // [N, packed_dim]
    int N, int d,
    int bit_width,
    int bytes_per_plane,
    int packed_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_bytes = N * packed_dim;
    if (idx >= total_bytes) return;

    int row = idx / packed_dim;
    int byte_out = idx % packed_dim;

    int plane = byte_out / bytes_per_plane;
    int byte_in_plane = byte_out % bytes_per_plane;
    int dim_start = byte_in_plane * 8;

    const uint8_t* codes_row = codes + (size_t)row * d;

    uint8_t packed_byte = 0;
    for (int b = 0; b < 8; b++) {
        int j = dim_start + b;
        if (j >= d) break;
        int bit_val = (codes_row[j] >> plane) & 1;
        packed_byte |= (bit_val << (7 - b));
    }

    packed[idx] = packed_byte;
}

torch::Tensor pack_codes_cuda(torch::Tensor codes, int64_t bit_width) {
    c10::cuda::CUDAGuard device_guard(codes.device());

    TORCH_CHECK(codes.dtype() == torch::kUInt8, "codes must be uint8");
    TORCH_CHECK(codes.dim() == 2, "codes must be 2D [N, d]");
    TORCH_CHECK(bit_width >= 1 && bit_width <= 4, "bit_width must be 1-4");

    auto codes_c = codes.contiguous();
    int N = codes_c.size(0);
    int d = codes_c.size(1);
    int bytes_per_plane = (d + 7) / 8;
    int packed_dim = bit_width * bytes_per_plane;

    auto packed = torch::zeros({N, packed_dim},
        torch::TensorOptions().dtype(torch::kUInt8).device(codes.device()));

    if (N == 0) return packed;

    int total_bytes = N * packed_dim;
    int threads = 256;
    int blocks = (total_bytes + threads - 1) / threads;

    pack_codes_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        codes_c.data_ptr<uint8_t>(),
        packed.data_ptr<uint8_t>(),
        N, d, bit_width, bytes_per_plane, packed_dim
    );

    return packed;
}

// -------------------------------------------------------------------
// Unpack kernel: each thread handles one (row, dimension) pair
// -------------------------------------------------------------------

__global__ void unpack_codes_kernel(
    const uint8_t* __restrict__ packed,  // [N, packed_dim]
    uint8_t* __restrict__ codes,         // [N, dim]
    int N, int dim,
    int bit_width,
    int bytes_per_plane,
    int packed_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * dim;
    if (idx >= total) return;

    int row = idx / dim;
    int j = idx % dim;

    const uint8_t* packed_row = packed + (size_t)row * packed_dim;

    int byte_idx = j / 8;
    int bit_pos = 7 - (j % 8);

    uint8_t code = 0;
    for (int i = 0; i < bit_width; i++) {
        uint8_t packed_byte = packed_row[i * bytes_per_plane + byte_idx];
        int bit_val = (packed_byte >> bit_pos) & 1;
        code |= (bit_val << i);
    }

    codes[idx] = code;
}

torch::Tensor unpack_codes_cuda(torch::Tensor packed, int64_t bit_width, int64_t dim) {
    c10::cuda::CUDAGuard device_guard(packed.device());

    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(packed.dim() == 2, "packed must be 2D");
    TORCH_CHECK(bit_width >= 1 && bit_width <= 4, "bit_width must be 1-4");

    auto packed_c = packed.contiguous();
    int N = packed_c.size(0);
    int bytes_per_plane = (dim + 7) / 8;
    int packed_dim = bit_width * bytes_per_plane;

    TORCH_CHECK(packed_c.size(1) == packed_dim,
                "packed dim mismatch: expected ", packed_dim, " got ", packed_c.size(1));

    auto codes = torch::zeros({N, dim},
        torch::TensorOptions().dtype(torch::kUInt8).device(packed.device()));

    if (N == 0) return codes;

    int total = N * dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    unpack_codes_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        packed_c.data_ptr<uint8_t>(),
        codes.data_ptr<uint8_t>(),
        N, dim, bit_width, bytes_per_plane, packed_dim
    );

    return codes;
}
