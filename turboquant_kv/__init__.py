"""
turboquant_kv — TurboQuant KV cache compression and vector search.

Faithful implementation of arXiv:2504.19874.
"""

from turboquant_kv.config import TurboQuantConfig
from turboquant_kv.cache import QuantizedKVCache
from turboquant_kv.search import TurboQuantIndex
from turboquant_kv.distributed import DistributedQuantizedKVCache
from turboquant_kv.entropy import (
    HuffmanCoder,
    EntropyPackedStorage,
    compute_codeword_probabilities,
)
from turboquant_kv.reference import (
    quantize_mse,
    dequantize_mse,
    quantize_prod,
    dequantize_prod,
)

# HuggingFace Transformers integration (requires transformers)
try:
    from turboquant_kv.hf_integration import TurboQuantCache
except ImportError:
    TurboQuantCache = None  # type: ignore[assignment,misc]

# vLLM plugin (requires vllm)
try:
    from turboquant_kv.vllm_plugin import (
        TurboQuantKVCacheConfig,
        TurboQuantPagedKVCache,
        TurboQuantAttentionBackend,
    )
except ImportError:
    TurboQuantKVCacheConfig = None  # type: ignore[assignment,misc]
    TurboQuantPagedKVCache = None  # type: ignore[assignment,misc]
    TurboQuantAttentionBackend = None  # type: ignore[assignment,misc]

__version__ = "0.1.0"

__all__ = [
    "TurboQuantConfig",
    "QuantizedKVCache",
    "TurboQuantIndex",
    "DistributedQuantizedKVCache",
    "HuffmanCoder",
    "EntropyPackedStorage",
    "compute_codeword_probabilities",
    "quantize_mse",
    "dequantize_mse",
    "quantize_prod",
    "dequantize_prod",
    # HF integration
    "TurboQuantCache",
    # vLLM plugin
    "TurboQuantKVCacheConfig",
    "TurboQuantPagedKVCache",
    "TurboQuantAttentionBackend",
]
