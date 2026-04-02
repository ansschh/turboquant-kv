"""TurboQuant configuration."""

from dataclasses import dataclass


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache compression.

    Attributes:
        key_bits: Quantization bit width for keys (default 4).
        value_bits: Quantization bit width for values (default 2).
        mode: Quantization mode, "mse" (Algorithm 1) or "prod" (Algorithm 2).
        rotation: Rotation method, "dense_qr" (exact paper) or "rht" (fast approx).
        outlier_channels: Number of channels to quantize at +1 bit.
        protected_layers: First N layers kept at full precision.
        quantize_on_append: Whether to quantize on append (True) or defer.
        block_size: 0 = per-token quantization, >0 = group quantization.
        seed: Random seed for rotation matrix generation.
    """

    key_bits: int = 4
    value_bits: int = 2
    mode: str = "mse"
    rotation: str = "dense_qr"
    outlier_channels: int = 0
    protected_layers: int = 0
    quantize_on_append: bool = True
    block_size: int = 0
    seed: int = 42
