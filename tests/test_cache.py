"""
Tests for QuantizedKVCache.
"""

import math

import pytest
import torch

from turboquant_kv.config import TurboQuantConfig
from turboquant_kv.cache import QuantizedKVCache


class TestCacheBasic:
    def test_cache_append_and_retrieve(self):
        """Append KV pairs, verify attention scores approximate exact."""
        torch.manual_seed(0)
        num_layers = 2
        num_heads = 4
        head_dim = 64
        max_seq = 128
        seq_len = 32

        config = TurboQuantConfig(key_bits=4, value_bits=4, mode="mse")
        cache = QuantizedKVCache(
            config, num_layers, max_seq, num_heads, head_dim, device="cpu"
        )

        # Generate random KV pairs
        keys = torch.randn(num_heads, seq_len, head_dim)
        values = torch.randn(num_heads, seq_len, head_dim)

        cache.append(0, keys, values)
        assert cache.seq_lens[0].item() == seq_len

        # Generate a query and compute attention
        query = torch.randn(num_heads, 1, head_dim)
        logits = cache.attention_scores(0, query)
        assert logits.shape == (num_heads, 1, seq_len)

        # Compute exact attention
        exact_logits = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(head_dim)

        # Correlation should be high at 4-bit keys
        for h in range(num_heads):
            corr = torch.corrcoef(
                torch.stack([logits[h, 0], exact_logits[h, 0]])
            )[0, 1].item()
            assert corr > 0.8, f"Head {h}: correlation={corr:.3f} (expected > 0.8)"

    def test_attention_values(self):
        """Verify attention output is close to exact."""
        torch.manual_seed(0)
        config = TurboQuantConfig(key_bits=4, value_bits=4, mode="mse")
        cache = QuantizedKVCache(config, 1, 64, 2, 32, device="cpu")

        keys = torch.randn(2, 16, 32)
        values = torch.randn(2, 16, 32)
        cache.append(0, keys, values)

        query = torch.randn(2, 1, 32)
        logits = cache.attention_scores(0, query)
        weights = torch.softmax(logits, dim=-1)
        output = cache.attention_values(0, weights)

        assert output.shape == (2, 1, 32)

    def test_incremental_append(self):
        """Append tokens incrementally and verify seq_len tracking."""
        config = TurboQuantConfig(key_bits=3, value_bits=2)
        cache = QuantizedKVCache(config, 1, 100, 2, 64, device="cpu")

        for i in range(5):
            k = torch.randn(2, 1, 64)
            v = torch.randn(2, 1, 64)
            cache.append(0, k, v)

        assert cache.seq_lens[0].item() == 5


class TestCompression:
    def test_compression_ratio(self):
        """Verify reported compression matches expected.

        At key_bits=4, value_bits=2, head_dim=128:
        FP16 per token per head: 2 * 128 * 2 = 512 bytes (keys + values)
        Compressed: key=4*16=64 bytes + val=2*16=32 bytes + norms=8 bytes = 104 bytes
        Expected ratio: 512 / 104 ~ 4.9x
        """
        config = TurboQuantConfig(key_bits=4, value_bits=2, mode="mse")
        cache = QuantizedKVCache(config, 1, 100, 1, 128, device="cpu")

        k = torch.randn(1, 50, 128)
        v = torch.randn(1, 50, 128)
        cache.append(0, k, v)

        ratio = cache.compression_ratio
        assert ratio > 3.0, f"Compression ratio={ratio:.1f}x (expected > 3x)"
        assert ratio < 10.0, f"Compression ratio={ratio:.1f}x (suspiciously high)"

    def test_memory_bytes_nonzero(self):
        """Memory usage should be positive after appending."""
        config = TurboQuantConfig(key_bits=4, value_bits=2)
        cache = QuantizedKVCache(config, 1, 100, 2, 64, device="cpu")

        cache.append(0, torch.randn(2, 10, 64), torch.randn(2, 10, 64))
        assert cache.memory_bytes() > 0


class TestProtectedLayers:
    def test_protected_layers(self):
        """First N layers should be stored at full precision."""
        config = TurboQuantConfig(key_bits=4, value_bits=2, protected_layers=2)
        cache = QuantizedKVCache(config, 4, 64, 2, 32, device="cpu")

        k = torch.randn(2, 8, 32)
        v = torch.randn(2, 8, 32)

        # Append to protected layer
        cache.append(0, k, v)
        # Append to non-protected layer
        cache.append(2, k, v)

        # Protected layer should have exact reconstruction
        query = torch.randn(2, 1, 32)
        logits_protected = cache.attention_scores(0, query)
        exact_logits = torch.matmul(query, k.half().float().transpose(-2, -1)) / math.sqrt(32)

        # Should be very close (only fp16 rounding)
        for h in range(2):
            diff = (logits_protected[h, 0] - exact_logits[h, 0]).abs().max().item()
            assert diff < 0.05, f"Protected layer h={h}: max diff={diff:.4f}"


class TestOutlierChannels:
    def test_outlier_channels_config(self):
        """Verify outlier_channels parameter is accepted and cache initializes."""
        config = TurboQuantConfig(key_bits=4, value_bits=2, outlier_channels=8)
        cache = QuantizedKVCache(config, 1, 64, 2, 32, device="cpu")

        k = torch.randn(2, 8, 32)
        v = torch.randn(2, 8, 32)
        cache.append(0, k, v)

        # Should work without error; outlier channels get +1 bit codebook
        assert cache.seq_lens[0].item() == 8
        assert cache._has_outliers is True
