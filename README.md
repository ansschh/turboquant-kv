# TurboQuant-KV

A faithful, production-quality implementation of the **TurboQuant** paper ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) for KV cache compression and vector search. TurboQuant provides near-optimal scalar quantization on the unit sphere via random rotation followed by Lloyd-Max quantization, with an unbiased inner-product estimator (TurboQuant-Prod) that uses QJL sign-bit correction to reduce quantization error for attention computations.

## Installation

```bash
pip install turboquant-kv
```

For development:

```bash
git clone https://github.com/your-org/turboquant-kv.git
cd turboquant-kv
pip install -e ".[dev]"
```

## Quick Start: KV Cache Compression

```python
import torch
from turboquant_kv import TurboQuantConfig, QuantizedKVCache

config = TurboQuantConfig(key_bits=4, value_bits=2, mode="mse")
cache = QuantizedKVCache(config, num_layers=32, max_seq_len=4096, num_heads=32, head_dim=128)
cache.append(layer_id=0, key=torch.randn(32, 1, 128), value=torch.randn(32, 1, 128))
logits = cache.attention_scores(layer_id=0, query=torch.randn(32, 1, 128))
```

## Quick Start: Vector Search

```python
import torch
from turboquant_kv import TurboQuantIndex

db = torch.randn(100000, 256)
index = TurboQuantIndex.from_vectors(db, bit_width=3, mode="mse")
scores, indices = index.search(torch.randn(10, 256), k=10)
```

## Benchmarks

| Setting | Compression | MSE Distortion | Top-10 Recall |
|---------|-------------|----------------|---------------|
| 4-bit MSE, d=128 | ~4x | TBD | TBD |
| 3-bit MSE, d=128 | ~5.3x | TBD | TBD |
| 2-bit MSE, d=128 | ~8x | TBD | TBD |

Run benchmarks:

```bash
python -m benchmarks.micro.bench_quantize
python -m benchmarks.quality.bench_distortion
```

## Citation

```bibtex
@article{turboquant2025,
  title={TurboQuant: Online Vector Quantization for Quantized KV-Cache and Beyond},
  author={...},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```

## License

Apache-2.0
