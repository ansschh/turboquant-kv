/*
 * bindings.cpp -- pybind11 bindings for the TurboQuant C++ core.
 *
 * Exposes:
 *   turboquant_kv._core.Codebook
 *   turboquant_kv._core.Index
 *   turboquant_kv._core.quantize()
 *   turboquant_kv._core.dequantize()
 *   turboquant_kv._core.pack()
 *   turboquant_kv._core.unpack()
 *   turboquant_kv._core.make_rotation()
 *   turboquant_kv._core.search_packed()
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "turboquant_core.h"

#include <memory>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;
using namespace turboquant;

// -----------------------------------------------------------------------
//  Helpers to validate numpy arrays
// -----------------------------------------------------------------------

static void check_contiguous_f32(const py::array_t<float>& a, const char* name) {
    if (!(a.flags() & py::array::c_style)) {
        throw std::invalid_argument(std::string(name) + " must be C-contiguous float32");
    }
}

static void check_contiguous_u8(const py::array_t<uint8_t>& a, const char* name) {
    if (!(a.flags() & py::array::c_style)) {
        throw std::invalid_argument(std::string(name) + " must be C-contiguous uint8");
    }
}

// -----------------------------------------------------------------------
//  Codebook wrapper
// -----------------------------------------------------------------------

class PyCodebook {
public:
    Codebook cb;

    PyCodebook(int bits, int dim) : cb(bits, dim) {}

    py::array_t<float> boundaries() const {
        int n = (int)cb.scaled_boundaries.size();
        auto arr = py::array_t<float>(n);
        auto buf = arr.mutable_unchecked<1>();
        for (int i = 0; i < n; i++) buf(i) = cb.scaled_boundaries[i];
        return arr;
    }

    py::array_t<float> centroids() const {
        int n = (int)cb.scaled_centroids.size();
        auto arr = py::array_t<float>(n);
        auto buf = arr.mutable_unchecked<1>();
        for (int i = 0; i < n; i++) buf(i) = cb.scaled_centroids[i];
        return arr;
    }

    int bits() const { return cb.bits; }
    int dim() const { return cb.dim; }
    int n_levels() const { return cb.n_levels; }
};

// -----------------------------------------------------------------------
//  Index wrapper
// -----------------------------------------------------------------------

class PyIndex {
public:
    std::unique_ptr<TurboQuantIndex> idx;

    PyIndex(int dim, int bit_width, int seed = 42)
        : idx(std::make_unique<TurboQuantIndex>(dim, bit_width, seed)) {}

    // Private constructor for load()
    PyIndex(std::unique_ptr<TurboQuantIndex> ptr) : idx(std::move(ptr)) {}

    void add(py::array_t<float, py::array::c_style | py::array::forcecast> vectors) {
        if (vectors.ndim() == 1) {
            if (vectors.shape(0) != idx->dim()) {
                throw std::invalid_argument("Vector dimension mismatch");
            }
            idx->add(vectors.data(), 1);
        } else if (vectors.ndim() == 2) {
            if (vectors.shape(1) != idx->dim()) {
                throw std::invalid_argument("Vector dimension mismatch");
            }
            idx->add(vectors.data(), (int)vectors.shape(0));
        } else {
            throw std::invalid_argument("vectors must be 1D or 2D");
        }
    }

    py::tuple search(py::array_t<float, py::array::c_style | py::array::forcecast> queries, int k = 10) {
        int Q;
        const float* qdata;

        if (queries.ndim() == 1) {
            if (queries.shape(0) != idx->dim()) {
                throw std::invalid_argument("Query dimension mismatch");
            }
            Q = 1;
            qdata = queries.data();
        } else if (queries.ndim() == 2) {
            if (queries.shape(1) != idx->dim()) {
                throw std::invalid_argument("Query dimension mismatch");
            }
            Q = (int)queries.shape(0);
            qdata = queries.data();
        } else {
            throw std::invalid_argument("queries must be 1D or 2D");
        }

        auto scores = py::array_t<float>({Q, k});
        auto ids = py::array_t<int64_t>({Q, k});

        idx->search(qdata, Q, k,
                     scores.mutable_data(), ids.mutable_data());

        return py::make_tuple(scores, ids);
    }

    void save(const std::string& path) { idx->save(path); }

    static PyIndex load_from_file(const std::string& path) {
        return PyIndex(TurboQuantIndex::load(path));
    }

    int dim() const { return idx->dim(); }
    int bit_width() const { return idx->bit_width(); }
    int seed() const { return idx->seed(); }
    int n_vectors() const { return idx->n_vectors(); }

    py::array_t<float> get_norms() const {
        int n = idx->n_vectors();
        auto arr = py::array_t<float>(n);
        std::memcpy(arr.mutable_data(), idx->norms().data(), n * sizeof(float));
        return arr;
    }

    py::array_t<uint8_t> get_packed_codes() const {
        int n = idx->n_vectors();
        int pdim = packed_dim_for(idx->dim(), idx->bit_width());
        auto arr = py::array_t<uint8_t>({n, pdim});
        std::memcpy(arr.mutable_data(), idx->packed_codes().data(),
                     (size_t)n * pdim);
        return arr;
    }
};

// -----------------------------------------------------------------------
//  Free functions
// -----------------------------------------------------------------------

static py::array_t<float> py_make_rotation(int dim, int seed = 42) {
    auto rot = make_rotation_matrix(dim, seed);
    auto arr = py::array_t<float>({dim, dim});
    std::memcpy(arr.mutable_data(), rot.data(), (size_t)dim * dim * sizeof(float));
    return arr;
}

static py::tuple py_quantize(
    py::array_t<float, py::array::c_style | py::array::forcecast> vectors,
    py::array_t<float, py::array::c_style | py::array::forcecast> rotation,
    py::array_t<float, py::array::c_style | py::array::forcecast> boundaries,
    int bit_width
) {
    if (vectors.ndim() != 2) throw std::invalid_argument("vectors must be 2D");
    if (rotation.ndim() != 2) throw std::invalid_argument("rotation must be 2D");

    int N = (int)vectors.shape(0);
    int dim = (int)vectors.shape(1);

    if (rotation.shape(0) != dim || rotation.shape(1) != dim) {
        throw std::invalid_argument("rotation must be (dim, dim)");
    }

    int n_levels = (int)boundaries.shape(0) + 1;
    if (n_levels != (1 << bit_width)) {
        throw std::invalid_argument("boundaries length must be 2^bit_width - 1");
    }

    // Build a temporary codebook from the provided boundaries
    Codebook cb;
    cb.bits = bit_width;
    cb.dim = dim;
    cb.n_levels = n_levels;
    cb.scale = 1.0 / std::sqrt((double)dim);
    cb.scaled_boundaries.assign(boundaries.data(), boundaries.data() + n_levels - 1);
    // We need centroids too -- compute from boundaries (midpoints of edges)
    // Actually for quantize we only need boundaries. But the Codebook struct
    // also needs centroids for dequantize. We'll compute them properly.
    // For quantize_vectors we only use scaled_boundaries.
    cb.scaled_centroids.resize(n_levels, 0.0f);  // dummy, not used in quantize

    int pdim = packed_dim_for(dim, bit_width);
    auto packed = py::array_t<uint8_t>({N, pdim});
    auto norms = py::array_t<float>(N);

    quantize_vectors(vectors.data(), N, dim,
                     rotation.data(), cb,
                     packed.mutable_data(), norms.mutable_data());

    return py::make_tuple(packed, norms);
}

static py::array_t<uint8_t> py_pack(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> codes,
    int bit_width
) {
    if (codes.ndim() != 2) throw std::invalid_argument("codes must be 2D");
    int N = (int)codes.shape(0);
    int dim = (int)codes.shape(1);
    int pdim = packed_dim_for(dim, bit_width);

    auto packed = py::array_t<uint8_t>({N, pdim});
    pack_codes(codes.data(), N, dim, bit_width, packed.mutable_data());
    return packed;
}

static py::array_t<uint8_t> py_unpack(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> packed,
    int bit_width,
    int dim
) {
    if (packed.ndim() != 2) throw std::invalid_argument("packed must be 2D");
    int N = (int)packed.shape(0);

    auto codes = py::array_t<uint8_t>({N, dim});
    unpack_codes(packed.data(), N, dim, bit_width, codes.mutable_data());
    return codes;
}

static py::array_t<float> py_dequantize(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> packed,
    py::array_t<float, py::array::c_style | py::array::forcecast> norms,
    int bit_width,
    int dim,
    py::array_t<float, py::array::c_style | py::array::forcecast> rotation,
    py::array_t<float, py::array::c_style | py::array::forcecast> centroids
) {
    if (packed.ndim() != 2) throw std::invalid_argument("packed must be 2D");
    int N = (int)packed.shape(0);

    Codebook cb;
    cb.bits = bit_width;
    cb.dim = dim;
    cb.n_levels = 1 << bit_width;
    cb.scale = 1.0 / std::sqrt((double)dim);
    cb.scaled_centroids.assign(centroids.data(), centroids.data() + cb.n_levels);
    // boundaries not needed for dequantize
    cb.scaled_boundaries.resize(cb.n_levels - 1, 0.0f);

    auto out = py::array_t<float>({N, dim});
    dequantize_vectors(packed.data(), norms.data(), N, dim,
                       rotation.data(), cb, out.mutable_data());
    return out;
}

static py::tuple py_search_packed(
    py::array_t<float, py::array::c_style | py::array::forcecast> queries,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> packed,
    py::array_t<float, py::array::c_style | py::array::forcecast> db_norms,
    int dim, int bit_width,
    py::array_t<float, py::array::c_style | py::array::forcecast> rotation,
    py::array_t<float, py::array::c_style | py::array::forcecast> centroids,
    int k
) {
    int Q;
    if (queries.ndim() == 1) {
        Q = 1;
    } else if (queries.ndim() == 2) {
        Q = (int)queries.shape(0);
    } else {
        throw std::invalid_argument("queries must be 1D or 2D");
    }

    int N = (int)packed.shape(0);

    auto scores = py::array_t<float>({Q, k});
    auto ids = py::array_t<int64_t>({Q, k});

    search_packed(queries.data(), Q,
                  packed.data(), db_norms.data(), N,
                  dim, bit_width,
                  rotation.data(),
                  centroids.data(),
                  k,
                  scores.mutable_data(), ids.mutable_data());

    return py::make_tuple(scores, ids);
}

// -----------------------------------------------------------------------
//  Module definition
// -----------------------------------------------------------------------

PYBIND11_MODULE(_core, m) {
    m.doc() = "TurboQuant C++ core -- fast quantization and search without PyTorch";

    // --- Codebook ---
    py::class_<PyCodebook>(m, "Codebook",
        "Lloyd-Max codebook for N(0, 1/sqrt(dim)) distribution.")
        .def(py::init<int, int>(), py::arg("bits"), py::arg("dim"),
             "Compute Lloyd-Max codebook for the given bit width and dimension.")
        .def_property_readonly("boundaries", &PyCodebook::boundaries,
             "Scaled boundaries as numpy float32 array, shape (2^bits - 1,).")
        .def_property_readonly("centroids", &PyCodebook::centroids,
             "Scaled centroids as numpy float32 array, shape (2^bits,).")
        .def_property_readonly("bits", &PyCodebook::bits)
        .def_property_readonly("dim", &PyCodebook::dim)
        .def_property_readonly("n_levels", &PyCodebook::n_levels)
        .def("__repr__", [](const PyCodebook& self) {
            std::ostringstream oss;
            oss << "Codebook(bits=" << self.bits() << ", dim=" << self.dim()
                << ", n_levels=" << self.n_levels() << ")";
            return oss.str();
        });

    // --- Index ---
    py::class_<PyIndex>(m, "Index",
        "TurboQuant compressed vector index with brute-force search.")
        .def(py::init<int, int, int>(),
             py::arg("dim"), py::arg("bit_width") = 4, py::arg("seed") = 42,
             "Create a new empty index.")
        .def("add", &PyIndex::add, py::arg("vectors"),
             "Add vectors to the index. vectors: numpy float32 array, shape (N, dim) or (dim,).")
        .def("search", &PyIndex::search, py::arg("queries"), py::arg("k") = 10,
             "Search for k nearest neighbors. Returns (scores, ids) numpy arrays.")
        .def("save", &PyIndex::save, py::arg("path"),
             "Save index to a binary file.")
        .def_static("load", &PyIndex::load_from_file, py::arg("path"),
             "Load index from a binary file.")
        .def_property_readonly("dim", &PyIndex::dim)
        .def_property_readonly("bit_width", &PyIndex::bit_width)
        .def_property_readonly("seed", &PyIndex::seed)
        .def_property_readonly("n_vectors", &PyIndex::n_vectors)
        .def_property_readonly("norms", &PyIndex::get_norms,
             "Database vector norms as numpy float32 array.")
        .def_property_readonly("packed_codes", &PyIndex::get_packed_codes,
             "Packed codes as numpy uint8 array, shape (n_vectors, packed_dim).")
        .def("__repr__", [](const PyIndex& self) {
            std::ostringstream oss;
            oss << "Index(dim=" << self.dim() << ", bit_width=" << self.bit_width()
                << ", n_vectors=" << self.n_vectors() << ", seed=" << self.seed() << ")";
            return oss.str();
        });

    // --- Free functions ---
    m.def("make_rotation", &py_make_rotation,
          py::arg("dim"), py::arg("seed") = 42,
          "Generate orthogonal rotation matrix via QR of random Gaussian. "
          "Returns numpy float32 array of shape (dim, dim).");

    m.def("quantize", &py_quantize,
          py::arg("vectors"), py::arg("rotation"), py::arg("boundaries"), py::arg("bit_width"),
          "Quantize vectors: normalize, rotate, searchsorted, bit-pack. "
          "Returns (packed, norms) numpy arrays.");

    m.def("dequantize", &py_dequantize,
          py::arg("packed"), py::arg("norms"), py::arg("bit_width"), py::arg("dim"),
          py::arg("rotation"), py::arg("centroids"),
          "Dequantize packed codes back to vectors. "
          "Returns numpy float32 array of shape (N, dim).");

    m.def("pack", &py_pack,
          py::arg("codes"), py::arg("bit_width"),
          "Pack uint8 codes into bit-plane format. "
          "Returns numpy uint8 array of shape (N, packed_dim).");

    m.def("unpack", &py_unpack,
          py::arg("packed"), py::arg("bit_width"), py::arg("dim"),
          "Unpack bit-plane packed codes. "
          "Returns numpy uint8 array of shape (N, dim).");

    m.def("search_packed", &py_search_packed,
          py::arg("queries"), py::arg("packed"), py::arg("db_norms"),
          py::arg("dim"), py::arg("bit_width"),
          py::arg("rotation"), py::arg("centroids"), py::arg("k") = 10,
          "Search packed database using precomputed lookup table. "
          "Returns (scores, ids) numpy arrays of shape (Q, k).");
}
