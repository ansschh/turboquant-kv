/*
 * registration.cpp — Register all TurboQuant ops with the PyTorch dispatcher.
 *
 * Defines the op schemas and binds CUDA + CPU implementations.
 */

#include <torch/extension.h>

// -------------------------------------------------------------------
// Forward declarations — CUDA kernels (from csrc/cuda/*.cu)
// -------------------------------------------------------------------

#ifdef WITH_CUDA

std::tuple<torch::Tensor, torch::Tensor> rotate_and_quantize_cuda(
    torch::Tensor vectors,
    torch::Tensor rotation,
    torch::Tensor boundaries,
    int64_t bit_width
);

torch::Tensor attention_scores_packed_cuda(
    torch::Tensor query,
    torch::Tensor packed_keys,
    torch::Tensor key_norms,
    torch::Tensor centroids,
    torch::Tensor rotation,
    int64_t bit_width
);

torch::Tensor attention_values_packed_cuda(
    torch::Tensor attn_weights,
    torch::Tensor packed_values,
    torch::Tensor value_norms,
    torch::Tensor centroids,
    torch::Tensor rotation_T,
    int64_t bit_width
);

torch::Tensor pack_codes_cuda(torch::Tensor codes, int64_t bit_width);

torch::Tensor unpack_codes_cuda(torch::Tensor packed, int64_t bit_width, int64_t dim);

#endif  // WITH_CUDA

// -------------------------------------------------------------------
// Forward declarations — CPU fallbacks (from csrc/cpu/quant_cpu.cpp)
// -------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> rotate_and_quantize_cpu(
    torch::Tensor vectors,
    torch::Tensor rotation,
    torch::Tensor boundaries,
    int64_t bit_width
);

torch::Tensor attention_scores_packed_cpu(
    torch::Tensor query,
    torch::Tensor packed_keys,
    torch::Tensor key_norms,
    torch::Tensor centroids,
    torch::Tensor rotation,
    int64_t bit_width
);

torch::Tensor attention_values_packed_cpu(
    torch::Tensor attn_weights,
    torch::Tensor packed_values,
    torch::Tensor value_norms,
    torch::Tensor centroids,
    torch::Tensor rotation_T,
    int64_t bit_width
);

torch::Tensor pack_codes_cpu(torch::Tensor codes, int64_t bit_width);

torch::Tensor unpack_codes_cpu(torch::Tensor packed, int64_t bit_width, int64_t dim);


// -------------------------------------------------------------------
// Schema definitions
// -------------------------------------------------------------------

TORCH_LIBRARY(turboquant, m) {
    m.def(
        "rotate_and_quantize(Tensor vectors, Tensor rotation, Tensor boundaries, int bit_width)"
        " -> (Tensor, Tensor)"
    );
    m.def(
        "attention_scores_packed(Tensor query, Tensor packed_keys, Tensor key_norms,"
        " Tensor centroids, Tensor rotation, int bit_width) -> Tensor"
    );
    m.def(
        "attention_values_packed(Tensor attn_weights, Tensor packed_values, Tensor value_norms,"
        " Tensor centroids, Tensor rotation_T, int bit_width) -> Tensor"
    );
    m.def("pack_codes(Tensor codes, int bit_width) -> Tensor");
    m.def("unpack_codes(Tensor packed, int bit_width, int dim) -> Tensor");
}


// -------------------------------------------------------------------
// CPU dispatch
// -------------------------------------------------------------------

TORCH_LIBRARY_IMPL(turboquant, CPU, m) {
    m.impl("rotate_and_quantize", &rotate_and_quantize_cpu);
    m.impl("attention_scores_packed", &attention_scores_packed_cpu);
    m.impl("attention_values_packed", &attention_values_packed_cpu);
    m.impl("pack_codes", &pack_codes_cpu);
    m.impl("unpack_codes", &unpack_codes_cpu);
}


// -------------------------------------------------------------------
// CUDA dispatch
// -------------------------------------------------------------------

#ifdef WITH_CUDA

TORCH_LIBRARY_IMPL(turboquant, CUDA, m) {
    m.impl("rotate_and_quantize", &rotate_and_quantize_cuda);
    m.impl("attention_scores_packed", &attention_scores_packed_cuda);
    m.impl("attention_values_packed", &attention_values_packed_cuda);
    m.impl("pack_codes", &pack_codes_cuda);
    m.impl("unpack_codes", &unpack_codes_cuda);
}

#endif  // WITH_CUDA


// -------------------------------------------------------------------
// Python module entry point
// -------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TurboQuant CUDA/CPU kernels";
}
