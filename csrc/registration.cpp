#include <torch/extension.h>

// Placeholder for CUDA kernel registration.
// Will register:
//   turboquant::rotate_and_quantize
//   turboquant::attention_scores_packed
//   turboquant::dequantize_values_blocked
//   turboquant::pack_codes
//   turboquant::unpack_codes

TORCH_LIBRARY(turboquant, m) {
    // Stubs - to be implemented in csrc/cuda/ and csrc/cpu/
    // m.def("rotate_and_quantize(Tensor x, Tensor rotation, Tensor boundaries) -> (Tensor, Tensor)");
    // m.def("attention_scores_packed(Tensor query, Tensor packed_keys, Tensor norms, Tensor centroids, Tensor rotation) -> Tensor");
    // m.def("dequantize_values_blocked(Tensor packed, Tensor norms, Tensor centroids, Tensor rotation, int block_size) -> Tensor");
}
