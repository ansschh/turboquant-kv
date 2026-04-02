#!/usr/bin/env python3
"""
TurboQuant KV Cache Demo -- End-to-end compression on a real LLM.

Loads a HuggingFace model, runs generation with full-precision and TurboQuant-
compressed KV caches, and reports memory savings, speed, and output quality.

Usage:
    python demos/llm_kv_cache_demo.py --model meta-llama/Llama-3.2-1B-Instruct
    python demos/llm_kv_cache_demo.py --model microsoft/phi-2
    python demos/llm_kv_cache_demo.py  # auto-selects a model
"""

from __future__ import annotations

import argparse
import gc
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Lazy imports -- fail gracefully with clear messages
# ---------------------------------------------------------------------------

def _check_deps():
    missing = []
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    try:
        import turboquant_kv  # noqa: F401
    except ImportError:
        missing.append("turboquant_kv")
    if missing:
        print(f"[ERROR] Missing dependencies: {', '.join(missing)}")
        print("Install with:  pip install transformers torch turboquant_kv")
        sys.exit(1)

_check_deps()

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from turboquant_kv.hf_integration import TurboQuantCache

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Models to try in order (smallest last as fallback)
DEFAULT_MODEL_PRIORITY = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
    "facebook/opt-1.3b",
    "facebook/opt-350m",
    "gpt2",
]

# Long-context prompt: a real multi-paragraph passage the model can continue
LONG_CONTEXT_SEED = textwrap.dedent("""\
    The history of artificial intelligence began in antiquity, with myths, stories \
    and rumors of artificial beings endowed with intelligence or consciousness by \
    master craftsmen. The seeds of modern AI were planted by philosophers who \
    attempted to describe the process of human thinking as the mechanical \
    manipulation of symbols. This work culminated in the invention of the \
    programmable digital computer in the 1940s, a machine based on the abstract \
    essence of mathematical reasoning. This device and the ideas behind it \
    inspired a handful of scientists to begin seriously discussing the possibility \
    of building an electronic brain.

    The field of AI research was founded at a workshop held on the campus of \
    Dartmouth College during the summer of 1956. Those who attended would become \
    the leaders of AI research for decades. Many of them predicted that a machine \
    as intelligent as a human being would exist in no more than a generation, and \
    they were given millions of dollars to make this vision come true. Eventually, \
    it became obvious that commercial developers and researchers had grossly \
    underestimated the difficulty of the project. In 1974, in response to the \
    criticism from James Lighthill and ongoing pressure from congress, the U.S. \
    and British governments cut off exploratory research in AI. The next few years \
    would later be called an "AI winter", a period when obtaining funding for AI \
    projects was difficult.

    In the early 1980s, AI research was revived by the commercial success of \
    expert systems, a form of AI program that simulated the knowledge and \
    analytical skills of human experts. By 1985, the market for AI had reached \
    over a billion dollars. At the same time, Japan's fifth generation computer \
    project inspired the U.S. and British governments to restore funding for \
    academic research. However, beginning with the collapse of the Lisp Machine \
    market in 1987, AI once again fell into disrepute, and a second, longer-lasting \
    winter began.

    AI research has been revived multiple times since then: in the late 1990s and \
    early 21st century by advances in machine learning and statistical methods; \
    and again after 2012 with breakthroughs in deep learning. The field achieved \
    commercial viability through applications ranging from medical diagnosis to \
    self-driving cars, natural language processing, game playing, and creative \
    applications like art and music generation.

    The development of large language models, beginning with GPT-2 in 2019 and \
    dramatically advancing with GPT-3 in 2020, opened a new chapter. These models \
    demonstrated emergent capabilities -- the ability to perform tasks they were \
    never explicitly trained on -- which sparked both excitement and concern among \
    researchers. The scale of these models grew rapidly: from millions to billions \
    to trillions of parameters, each generation pushing the boundaries of what \
    artificial intelligence could achieve.

    Transformer architectures, introduced by Vaswani et al. in the landmark \
    "Attention Is All You Need" paper of 2017, became the backbone of modern AI \
    systems. The key innovation was the self-attention mechanism, which allowed \
    models to weigh the importance of different parts of the input when producing \
    each part of the output. This mechanism, while powerful, came with a \
    significant cost: the key-value cache required during autoregressive generation \
    grows linearly with sequence length, consuming substantial GPU memory.

    This memory bottleneck has motivated a rich line of research into KV cache \
    compression techniques. Methods like quantization, eviction, and attention \
    sparsity all aim to reduce the memory footprint without degrading generation \
    quality. TurboQuant represents a principled approach to this problem, using \
    random rotation followed by optimal scalar quantization to compress keys and \
    values with minimal information loss.\
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Stores results from a single generation run."""
    config_name: str
    kv_memory_bytes: int
    fp16_baseline_bytes: int  # theoretical full fp16 KV size
    compression_ratio: float
    tokens_per_sec: float
    peak_gpu_mb: float
    generated_text: str
    num_context_tokens: int
    num_generated_tokens: int


def _gpu_mem_mb() -> float:
    """Current GPU memory allocated in MB (0.0 if no CUDA)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def _peak_gpu_mem_mb() -> float:
    """Peak GPU memory allocated in MB (0.0 if no CUDA)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def _reset_peak():
    """Reset CUDA peak memory counter."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _cleanup():
    """Force garbage collection and free CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _compute_fp16_kv_bytes(model_config, seq_len: int) -> int:
    """Estimate FP16 KV cache size in bytes for a given sequence length.

    KV cache = 2 (K+V) * num_layers * num_kv_heads * head_dim * seq_len * 2 (fp16 bytes)
    """
    num_layers = model_config.num_hidden_layers
    # Handle GQA: num_key_value_heads may be less than num_attention_heads
    num_kv_heads = getattr(model_config, "num_key_value_heads",
                           model_config.num_attention_heads)
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * 2


def _format_bytes(n: int) -> str:
    """Human-readable byte count."""
    if n >= 1024 ** 3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _build_long_input(tokenizer, target_tokens: int = 4096) -> Tuple[torch.Tensor, str]:
    """Build a long-context input by repeating the seed text.

    Returns (input_ids [1, seq_len], raw_text).
    """
    # Tokenize the seed once to measure length
    seed_ids = tokenizer.encode(LONG_CONTEXT_SEED, add_special_tokens=False)
    seed_len = len(seed_ids)

    # Repeat seed text enough times
    reps = max(1, (target_tokens // seed_len) + 1)
    full_text = (LONG_CONTEXT_SEED + "\n\n") * reps

    # Tokenize and truncate
    ids = tokenizer.encode(full_text, add_special_tokens=True,
                           truncation=True, max_length=target_tokens,
                           return_tensors="pt")
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor([ids])
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids, full_text


def _try_load_wikitext(tokenizer, target_tokens: int = 4096) -> Optional[Tuple[torch.Tensor, str]]:
    """Try loading wikitext-103 for a real long-context input."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test",
                          trust_remote_code=True)
        # Concatenate paragraphs until we have enough tokens
        texts = []
        total_chars = 0
        # Rough estimate: 4 chars per token
        target_chars = target_tokens * 5
        for row in ds:
            text = row["text"].strip()
            if len(text) < 20:
                continue
            texts.append(text)
            total_chars += len(text)
            if total_chars >= target_chars:
                break
        full_text = "\n\n".join(texts)
        ids = tokenizer.encode(full_text, add_special_tokens=True,
                               truncation=True, max_length=target_tokens,
                               return_tensors="pt")
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor([ids])
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.shape[1] >= target_tokens // 2:
            return ids, full_text
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Generation with different cache types
# ---------------------------------------------------------------------------

def run_generation(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    cache_type: str,
    max_new_tokens: int = 128,
    key_bits: int = 4,
    value_bits: int = 2,
) -> RunResult:
    """Run generation and measure metrics.

    Args:
        cache_type: "full", "tq" (TurboQuant)
        key_bits, value_bits: only used when cache_type == "tq"
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]

    config_name = {
        "full": "Full Precision",
    }.get(cache_type, f"TQ K{key_bits}/V{value_bits}")

    _cleanup()
    _reset_peak()

    # Build the cache object
    if cache_type == "full":
        cache = DynamicCache()
    else:
        cache = TurboQuantCache(
            key_bits=key_bits,
            value_bits=value_bits,
            mode="mse",
            rotation="dense_qr",
            protected_layers=0,
            seed=42,
        )

    # Warm up (first token often slower due to CUDA graph compilation)
    # We skip warm-up to keep the demo simple and honest.

    # Generate
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            temperature=1.0,
            use_cache=True,
        )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()

    elapsed = t1 - t0
    num_generated = outputs.shape[1] - seq_len
    tok_per_sec = num_generated / elapsed if elapsed > 0 else 0.0

    peak_gpu = _peak_gpu_mem_mb()

    # Measure KV cache memory
    if cache_type == "full":
        # For DynamicCache, estimate from model config
        total_seq = seq_len + num_generated
        kv_bytes = _compute_fp16_kv_bytes(model.config, total_seq)
    else:
        kv_bytes = cache.memory_bytes()

    fp16_bytes = _compute_fp16_kv_bytes(model.config, seq_len + num_generated)
    ratio = fp16_bytes / kv_bytes if kv_bytes > 0 else 1.0

    # Decode generated text
    gen_ids = outputs[0, seq_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return RunResult(
        config_name=config_name,
        kv_memory_bytes=kv_bytes,
        fp16_baseline_bytes=fp16_bytes,
        compression_ratio=ratio,
        tokens_per_sec=tok_per_sec,
        peak_gpu_mb=peak_gpu,
        generated_text=gen_text,
        num_context_tokens=seq_len,
        num_generated_tokens=num_generated,
    )


# ---------------------------------------------------------------------------
# Output comparison
# ---------------------------------------------------------------------------

def compare_outputs(results: List[RunResult], max_chars: int = 200):
    """Print side-by-side output comparison."""
    print("\n--- Output Comparison (first {} chars) ---\n".format(max_chars))
    baseline = results[0].generated_text
    for r in results:
        tag = f"[{r.config_name}]"
        snippet = r.generated_text[:max_chars]
        # Replace newlines for cleaner display
        snippet = snippet.replace("\n", " ")
        print(f"  {tag:20s} {snippet}")

    # Check for divergence
    print()
    for r in results[1:]:
        if r.generated_text == baseline:
            print(f"  {r.config_name}: EXACT MATCH with baseline")
        else:
            # Find first divergence point
            min_len = min(len(baseline), len(r.generated_text))
            diverge_at = min_len
            for i in range(min_len):
                if baseline[i] != r.generated_text[i]:
                    diverge_at = i
                    break
            print(f"  {r.config_name}: diverges at char {diverge_at}"
                  f" (of {len(baseline)} baseline / {len(r.generated_text)} compressed)")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: List[RunResult], model_name: str):
    """Print a clean summary table."""
    if not results:
        return

    r0 = results[0]
    ctx = r0.num_context_tokens
    gen = r0.num_generated_tokens

    header = f"TurboQuant KV Cache Demo -- {model_name}"
    print()
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    print(f"Context: {ctx} tokens | Generated: {gen} tokens")
    print()

    # Table
    col_w = [17, 12, 13, 8, 10]
    hdr = ["Config", "KV Memory", "Compression", "tok/s", "Peak GPU"]
    sep = ["-" * w for w in col_w]
    fmt_row = "| {:<17s} | {:>10s} | {:>11s} | {:>6s} | {:>8s} |"

    print(fmt_row.format(*hdr))
    print(fmt_row.format(*sep))
    for r in results:
        print(fmt_row.format(
            r.config_name,
            _format_bytes(r.kv_memory_bytes),
            f"{r.compression_ratio:.1f}x",
            f"{r.tokens_per_sec:.1f}",
            f"{r.peak_gpu_mb:.0f} MB" if r.peak_gpu_mb > 0 else "N/A",
        ))
    print()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: Optional[str] = None) -> Tuple[str, object, object]:
    """Load model and tokenizer. Returns (model_name, model, tokenizer).

    Tries the specified model first, then falls back through the priority list.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    candidates = [model_name] if model_name else []
    candidates.extend(DEFAULT_MODEL_PRIORITY)

    last_error = None
    for name in candidates:
        if name is None:
            continue
        try:
            print(f"[INFO] Loading {name} ...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            if device == "cpu":
                model = model.to(device)
            model.eval()

            # Ensure pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print(f"[INFO] Loaded {name} on {device} ({dtype})")
            return name, model, tokenizer
        except Exception as e:
            last_error = e
            print(f"[WARN] Could not load {name}: {e}")
            _cleanup()
            continue

    print(f"[ERROR] Failed to load any model. Last error: {last_error}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant KV Cache compression demo on a real LLM."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model ID (default: auto-select from priority list)"
    )
    parser.add_argument(
        "--context-tokens", type=int, default=4096,
        help="Target context length in tokens (default: 4096)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128,
        help="Number of tokens to generate (default: 128)"
    )
    parser.add_argument(
        "--no-wikitext", action="store_true",
        help="Skip attempting to load wikitext dataset"
    )
    args = parser.parse_args()

    # Load model
    model_name, model, tokenizer = load_model(args.model)

    # Build long-context input
    print(f"[INFO] Building long-context input (~{args.context_tokens} tokens) ...")
    input_ids = None
    if not args.no_wikitext:
        result = _try_load_wikitext(tokenizer, target_tokens=args.context_tokens)
        if result is not None:
            input_ids, _ = result
            print(f"[INFO] Loaded wikitext-103 context: {input_ids.shape[1]} tokens")

    if input_ids is None:
        input_ids, _ = _build_long_input(tokenizer, target_tokens=args.context_tokens)
        print(f"[INFO] Built repeated-essay context: {input_ids.shape[1]} tokens")

    # Define configurations to test
    configs = [
        ("full", 0, 0),       # Full precision baseline
        ("tq", 4, 2),         # TurboQuant K4/V2
        ("tq", 3, 2),         # TurboQuant K3/V2
    ]

    results: List[RunResult] = []

    for cache_type, k_bits, v_bits in configs:
        label = "Full Precision" if cache_type == "full" else f"TQ K{k_bits}/V{v_bits}"
        print(f"\n[RUN] {label} ...", flush=True)
        try:
            r = run_generation(
                model, tokenizer, input_ids,
                cache_type=cache_type,
                max_new_tokens=args.max_new_tokens,
                key_bits=k_bits,
                value_bits=v_bits,
            )
            results.append(r)
            print(f"  -> {r.tokens_per_sec:.1f} tok/s, "
                  f"KV={_format_bytes(r.kv_memory_bytes)}, "
                  f"ratio={r.compression_ratio:.1f}x")
        except torch.cuda.OutOfMemoryError:
            print(f"  -> OOM! Skipping {label}.")
            _cleanup()
        except Exception as e:
            print(f"  -> Error: {e}")
            _cleanup()

    if not results:
        print("[ERROR] No successful runs. Exiting.")
        sys.exit(1)

    # Print summary
    print_summary(results, model_name)
    compare_outputs(results)

    # Optional: simple perplexity estimate on a held-out snippet
    _try_perplexity(model, tokenizer, results, model_name)


def _try_perplexity(model, tokenizer, results: List[RunResult], model_name: str):
    """Estimate perplexity on a short held-out text (optional quality metric).

    This measures the model's next-token prediction quality after a long context
    has been processed through each cache type. We compare the log-likelihoods
    of a short continuation.
    """
    print("\n--- Perplexity Comparison (experimental) ---\n")

    held_out = (
        "The transformer architecture revolutionized natural language processing "
        "by introducing the self-attention mechanism. Unlike recurrent neural networks, "
        "transformers can process all tokens in parallel during training, leading to "
        "massive speedups on modern hardware."
    )

    held_out_ids = tokenizer.encode(held_out, add_special_tokens=False,
                                    return_tensors="pt")
    if not isinstance(held_out_ids, torch.Tensor):
        held_out_ids = torch.tensor([held_out_ids])
    if held_out_ids.dim() == 1:
        held_out_ids = held_out_ids.unsqueeze(0)

    device = next(model.parameters()).device
    held_out_ids = held_out_ids.to(device)
    num_eval_tokens = held_out_ids.shape[1]

    if num_eval_tokens < 5:
        print("  Held-out text too short, skipping perplexity.")
        return

    configs = [
        ("Full Precision", "full", 0, 0),
        ("TQ K4/V2", "tq", 4, 2),
        ("TQ K3/V2", "tq", 3, 2),
    ]

    # Use a shorter context for perplexity to keep it fast
    short_ctx, _ = _build_long_input(tokenizer, target_tokens=512)
    short_ctx = short_ctx.to(device)

    for label, cache_type, k_bits, v_bits in configs:
        try:
            if cache_type == "full":
                cache = DynamicCache()
            else:
                cache = TurboQuantCache(
                    key_bits=k_bits, value_bits=v_bits,
                    mode="mse", rotation="dense_qr", seed=42,
                )

            # Prefill the cache with context
            with torch.no_grad():
                _ = model(short_ctx, past_key_values=cache, use_cache=True)

            # Now evaluate log-likelihood of held-out tokens
            combined = held_out_ids
            with torch.no_grad():
                out = model(combined, past_key_values=cache, use_cache=True)
                logits = out.logits  # (1, seq, vocab)

            # Shift: predict token t+1 from position t
            shift_logits = logits[:, :-1, :].float()
            shift_labels = combined[:, 1:]
            loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
            nll = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)),
                          shift_labels.reshape(-1))
            ppl = torch.exp(nll).item()

            print(f"  {label:20s}  perplexity = {ppl:.2f}")
            _cleanup()
        except Exception as e:
            print(f"  {label:20s}  error: {e}")
            _cleanup()

    print()


if __name__ == "__main__":
    main()
