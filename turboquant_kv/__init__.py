"""
turboquant_kv — TurboQuant KV cache compression and vector search.

Faithful implementation of arXiv:2504.19874.
"""

from turboquant_kv.config import TurboQuantConfig
from turboquant_kv.cache import QuantizedKVCache
from turboquant_kv.search import TurboQuantIndex
from turboquant_kv.reference import (
    quantize_mse,
    dequantize_mse,
    quantize_prod,
    dequantize_prod,
)

__version__ = "0.1.0"

__all__ = [
    "TurboQuantConfig",
    "QuantizedKVCache",
    "TurboQuantIndex",
    "quantize_mse",
    "dequantize_mse",
    "quantize_prod",
    "dequantize_prod",
]
