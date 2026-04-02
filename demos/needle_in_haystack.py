#!/usr/bin/env python3
"""
Needle-in-a-Haystack Test for TurboQuant KV Cache Compression.

Evaluates whether KV cache quantization degrades a model's ability to retrieve
specific facts buried at various depths in a long context.

Usage:
    python demos/needle_in_haystack.py --model meta-llama/Llama-3.2-1B-Instruct
    python demos/needle_in_haystack.py --model meta-llama/Llama-3.2-1B-Instruct --depths 0 50 100
    python demos/needle_in_haystack.py  # auto-select model
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Dependency check
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

DEFAULT_MODEL_PRIORITY = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
    "facebook/opt-1.3b",
    "facebook/opt-350m",
    "gpt2",
]

# The needle: a distinctive fact that is easy to verify
NEEDLE = "The secret code to access the vault is: BLUE-TIGER-42."
NEEDLE_ANSWER = "BLUE-TIGER-42"

# The retrieval question
RETRIEVAL_PROMPT = "\n\nQuestion: What is the secret code to access the vault? Answer:"

# Filler paragraphs (real-ish text about diverse topics to pad the haystack)
FILLER_PARAGRAPHS = [
    "The Amazon rainforest covers approximately 5.5 million square kilometers "
    "and is home to an estimated 10 percent of all species on Earth. The forest "
    "produces roughly 20 percent of the world's oxygen and contains the largest "
    "collection of living plant and animal species in the world. Deforestation "
    "rates have varied considerably over the past several decades, with recent "
    "years seeing both progress and setbacks in conservation efforts.",

    "The Great Wall of China stretches over 13,000 miles across northern China. "
    "Construction began in the 7th century BC and continued for more than two "
    "millennia. The wall was built to protect Chinese states and empires against "
    "raids and invasions from various nomadic groups. Today it is one of the "
    "most popular tourist attractions in the world, receiving millions of "
    "visitors annually.",

    "Photosynthesis is the process by which green plants and certain other "
    "organisms transform light energy into chemical energy. During photosynthesis, "
    "carbon dioxide and water are converted into glucose and oxygen. The process "
    "occurs primarily in the leaves of plants, within specialized organelles "
    "called chloroplasts. Without photosynthesis, the atmosphere would not "
    "contain enough oxygen to support most forms of life.",

    "The human genome contains approximately 3 billion base pairs of DNA, "
    "organized into 23 pairs of chromosomes. The Human Genome Project, completed "
    "in 2003, successfully mapped all the genes in the human genome. Since then, "
    "advances in sequencing technology have dramatically reduced the cost and "
    "time required to sequence an individual's genome, opening new possibilities "
    "for personalized medicine.",

    "Jupiter is the largest planet in our solar system, with a mass more than "
    "twice that of all other planets combined. It has a Great Red Spot, a storm "
    "larger than Earth that has been raging for at least 400 years. Jupiter has "
    "at least 95 known moons, including the four large Galilean moons discovered "
    "by Galileo Galilei in 1610: Io, Europa, Ganymede, and Callisto.",

    "The theory of general relativity, published by Albert Einstein in 1915, "
    "describes gravity as a geometric property of space and time. According to "
    "this theory, massive objects cause a distortion in space-time, which is "
    "felt as gravity. General relativity has been confirmed by numerous "
    "experiments and observations, including the detection of gravitational "
    "waves by the LIGO observatory in 2015.",

    "The Industrial Revolution, which began in Britain in the late 18th century, "
    "transformed economies that had been based on agriculture and handicrafts "
    "into economies based on large-scale industry and mechanized manufacturing. "
    "Key innovations included the steam engine, the spinning jenny, and the "
    "power loom. The revolution spread throughout Europe and North America "
    "during the 19th century.",

    "Quantum mechanics is a fundamental theory in physics that describes the "
    "behavior of matter and energy at the atomic and subatomic levels. Unlike "
    "classical mechanics, quantum mechanics introduces concepts such as wave-"
    "particle duality, superposition, and entanglement. The theory has been "
    "extraordinarily successful in explaining a wide range of phenomena.",

    "The Mediterranean diet is characterized by high consumption of olive oil, "
    "fruits, vegetables, legumes, and whole grains, moderate consumption of "
    "fish and dairy, and low consumption of red meat. Numerous studies have "
    "shown that this dietary pattern is associated with reduced risk of "
    "cardiovascular disease, diabetes, and certain cancers.",

    "Antarctica is Earth's southernmost continent and contains about 70 percent "
    "of the planet's fresh water, locked in its ice sheet. The Antarctic ice "
    "sheet is the largest single mass of ice on Earth. If it were to melt "
    "entirely, global sea levels would rise approximately 58 meters. The "
    "continent has no permanent human population but hosts research stations "
    "operated by over 30 nations.",

    "Machine learning is a subset of artificial intelligence that focuses on "
    "building systems that learn from data. Unlike traditional programming, "
    "where rules are explicitly coded, machine learning algorithms identify "
    "patterns in data and use those patterns to make predictions or decisions. "
    "Common techniques include supervised learning, unsupervised learning, "
    "and reinforcement learning.",

    "The Mariana Trench in the western Pacific Ocean is the deepest known "
    "part of the world's oceans, reaching a depth of about 11,034 meters at "
    "its lowest point, the Challenger Deep. The trench is approximately 2,550 "
    "kilometers long and 69 kilometers wide. Despite the extreme pressure and "
    "absence of light, various forms of life have been discovered at these "
    "extraordinary depths.",

    "The Renaissance was a cultural and intellectual movement that began in "
    "Italy in the 14th century and spread throughout Europe over the next "
    "three centuries. It was characterized by a renewed interest in classical "
    "Greek and Roman thought, advances in art and architecture, the development "
    "of perspective in painting, and significant progress in science and "
    "exploration.",

    "Coral reefs cover less than 1 percent of the ocean floor but support "
    "about 25 percent of all marine species. They are built by colonies of "
    "tiny animals called coral polyps, which secrete calcium carbonate to form "
    "hard, protective structures. Coral reefs are under severe threat from "
    "climate change, ocean acidification, pollution, and destructive fishing "
    "practices.",

    "The Silk Road was a network of trade routes connecting the East and West "
    "for centuries. It was central to cultural interaction through regions of "
    "the Asian continent connecting the East and West by merchants, pilgrims, "
    "monks, soldiers, nomads, and urban dwellers from China to the Mediterranean "
    "Sea during various periods of history.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _try_load_wikitext_paragraphs(n_paragraphs: int = 200) -> Optional[List[str]]:
    """Try loading real paragraphs from wikitext-103."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test",
                          trust_remote_code=True)
        paragraphs = []
        for row in ds:
            text = row["text"].strip()
            if len(text) >= 100:  # skip short/header lines
                paragraphs.append(text)
            if len(paragraphs) >= n_paragraphs:
                break
        if len(paragraphs) >= 20:
            return paragraphs
    except Exception:
        pass
    return None


def _build_haystack(
    tokenizer,
    target_tokens: int,
    needle: str,
    depth_pct: float,
    paragraphs: List[str],
) -> Tuple[str, torch.Tensor, bool]:
    """Build a haystack with a needle inserted at a given depth.

    Args:
        target_tokens: approximate total context length in tokens
        needle: the fact string to insert
        depth_pct: 0.0 = beginning, 1.0 = end
        paragraphs: list of filler paragraphs

    Returns:
        (full_text, input_ids [1, seq_len])
    """
    # Build enough filler paragraphs
    filler_parts = []
    idx = 0
    # We cycle through paragraphs
    while True:
        filler_parts.append(paragraphs[idx % len(paragraphs)])
        idx += 1
        # Check approximate token count
        joined = "\n\n".join(filler_parts)
        approx_tokens = len(tokenizer.encode(joined, add_special_tokens=False))
        if approx_tokens >= target_tokens:
            break
        if idx > 10000:  # safety valve
            break

    n_parts = len(filler_parts)

    # Determine insertion point
    insert_idx = max(0, min(int(depth_pct * n_parts), n_parts))

    # Insert needle
    filler_parts.insert(insert_idx, needle)

    # Append retrieval prompt at the end
    full_text = "\n\n".join(filler_parts) + RETRIEVAL_PROMPT

    # Tokenize and truncate (leave room for generation)
    ids = tokenizer.encode(full_text, add_special_tokens=True,
                           truncation=True, max_length=target_tokens,
                           return_tensors="pt")
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor([ids])
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)

    # Verify the needle is actually in the tokenized text
    decoded = tokenizer.decode(ids[0], skip_special_tokens=True)
    needle_present = NEEDLE_ANSWER in decoded

    return full_text, ids, needle_present


@dataclass
class TrialResult:
    context_len: int
    depth_pct: int  # percentage
    config_name: str
    passed: bool
    generated_text: str
    needle_present_in_context: bool
    elapsed_sec: float


def run_trial(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    cache_type: str,
    key_bits: int = 4,
    value_bits: int = 2,
    max_new_tokens: int = 64,
) -> Tuple[str, bool]:
    """Run a single needle retrieval trial.

    Returns (generated_text, passed).
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

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

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
        )

    gen_ids = outputs[0, input_ids.shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Check if the answer is in the generated text
    passed = NEEDLE_ANSWER.lower() in gen_text.lower()

    return gen_text, passed


# ---------------------------------------------------------------------------
# Model loading (shared with llm_kv_cache_demo.py)
# ---------------------------------------------------------------------------

def load_model(model_name: Optional[str] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    candidates = [model_name] if model_name else []
    candidates.extend(DEFAULT_MODEL_PRIORITY)

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
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"[INFO] Loaded {name} on {device} ({dtype})")
            return name, model, tokenizer
        except Exception as e:
            print(f"[WARN] Could not load {name}: {e}")
            _cleanup()

    print("[ERROR] Failed to load any model.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Needle-in-a-haystack test for TurboQuant KV cache compression."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--context-lengths", type=int, nargs="+", default=[1024, 2048, 4096],
        help="Context lengths to test (default: 1024 2048 4096)"
    )
    parser.add_argument(
        "--depths", type=int, nargs="+", default=[0, 25, 50, 75, 100],
        help="Needle depth percentages (default: 0 25 50 75 100)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Max tokens to generate per trial (default: 64)"
    )
    parser.add_argument(
        "--no-wikitext", action="store_true",
        help="Use built-in filler instead of wikitext"
    )
    args = parser.parse_args()

    model_name, model, tokenizer = load_model(args.model)

    # Load filler paragraphs
    paragraphs = None
    if not args.no_wikitext:
        paragraphs = _try_load_wikitext_paragraphs(n_paragraphs=300)
        if paragraphs:
            print(f"[INFO] Loaded {len(paragraphs)} paragraphs from wikitext-103")

    if paragraphs is None:
        paragraphs = FILLER_PARAGRAPHS
        print(f"[INFO] Using {len(paragraphs)} built-in filler paragraphs")

    # Define cache configurations
    cache_configs = [
        ("Full Precision", "full", 0, 0),
        ("TQ K4/V2", "tq", 4, 2),
        ("TQ K3/V2", "tq", 3, 2),
        ("TQ K2/V2", "tq", 2, 2),
    ]

    all_results: List[TrialResult] = []

    total_trials = len(args.context_lengths) * len(args.depths) * len(cache_configs)
    trial_num = 0

    for ctx_len in args.context_lengths:
        for depth in args.depths:
            depth_frac = depth / 100.0

            # Build haystack once per (ctx_len, depth)
            _, input_ids, needle_present = _build_haystack(
                tokenizer, ctx_len, NEEDLE, depth_frac, paragraphs
            )
            actual_len = input_ids.shape[1]

            if not needle_present:
                print(f"[WARN] Needle was truncated at ctx={ctx_len}, depth={depth}%. "
                      f"Actual tokens: {actual_len}. Marking as N/A.")

            for config_name, cache_type, k_bits, v_bits in cache_configs:
                trial_num += 1
                print(f"  [{trial_num}/{total_trials}] ctx={actual_len} "
                      f"depth={depth}% {config_name} ...", end="", flush=True)

                try:
                    t0 = time.perf_counter()
                    gen_text, passed = run_trial(
                        model, tokenizer, input_ids,
                        cache_type=cache_type,
                        key_bits=k_bits,
                        value_bits=v_bits,
                        max_new_tokens=args.max_new_tokens,
                    )
                    elapsed = time.perf_counter() - t0

                    # If needle was not in context, mark as N/A regardless
                    if not needle_present:
                        passed = False

                    result = TrialResult(
                        context_len=actual_len,
                        depth_pct=depth,
                        config_name=config_name,
                        passed=passed,
                        generated_text=gen_text,
                        needle_present_in_context=needle_present,
                        elapsed_sec=elapsed,
                    )
                    all_results.append(result)

                    status = "PASS" if passed else ("N/A" if not needle_present else "FAIL")
                    print(f" {status} ({elapsed:.1f}s)")

                except torch.cuda.OutOfMemoryError:
                    print(" OOM")
                    all_results.append(TrialResult(
                        context_len=actual_len, depth_pct=depth,
                        config_name=config_name, passed=False,
                        generated_text="OOM",
                        needle_present_in_context=needle_present,
                        elapsed_sec=0,
                    ))
                    _cleanup()
                except Exception as e:
                    print(f" ERROR: {e}")
                    all_results.append(TrialResult(
                        context_len=actual_len, depth_pct=depth,
                        config_name=config_name, passed=False,
                        generated_text=f"ERROR: {e}",
                        needle_present_in_context=needle_present,
                        elapsed_sec=0,
                    ))
                    _cleanup()

                _cleanup()

    # Print results table
    _print_results_table(all_results, cache_configs, args.context_lengths,
                         args.depths, model_name)

    # Print detailed failures
    _print_failures(all_results)


def _print_results_table(
    results: List[TrialResult],
    cache_configs: List[Tuple],
    context_lengths: List[int],
    depths: List[int],
    model_name: str,
):
    """Print the main results table."""
    config_names = [c[0] for c in cache_configs]

    header = f"Needle-in-a-Haystack Results -- {model_name}"
    print()
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    print(f"Needle: \"{NEEDLE}\"")
    print(f"Query:  \"{RETRIEVAL_PROMPT.strip()}\"")
    print()

    # Build lookup
    lookup: Dict[Tuple[int, int, str], TrialResult] = {}
    for r in results:
        lookup[(r.context_len, r.depth_pct, r.config_name)] = r

    # Determine actual context lengths used
    actual_ctx_by_target: Dict[int, int] = {}
    for r in results:
        for tgt in context_lengths:
            if abs(r.context_len - tgt) < tgt * 0.3:
                actual_ctx_by_target[tgt] = r.context_len
                break

    # Column widths
    config_col_w = max(16, max(len(c) for c in config_names) + 2)
    fmt_header = f"| {{:<7s}} | {{:<5s}} |" + " | ".join(
        f"{{:^{config_col_w}s}}" for _ in config_names
    ) + " |"
    fmt_sep = f"|{'-' * 9}|{'-' * 7}|" + "|".join(
        f"{'-' * (config_col_w + 2)}" for _ in config_names
    ) + "|"
    fmt_row = f"| {{:<7s}} | {{:<5s}} |" + " | ".join(
        f"{{:^{config_col_w}s}}" for _ in config_names
    ) + " |"

    print(fmt_header.format("Context", "Depth", *config_names))
    print(fmt_sep)

    for tgt_ctx in context_lengths:
        actual_ctx = actual_ctx_by_target.get(tgt_ctx, tgt_ctx)
        for depth in depths:
            row_values = []
            for cname in config_names:
                r = lookup.get((actual_ctx, depth, cname))
                if r is None:
                    # Try approximate match
                    for key, val in lookup.items():
                        if key[1] == depth and key[2] == cname and abs(key[0] - tgt_ctx) < tgt_ctx * 0.3:
                            r = val
                            break

                if r is None:
                    row_values.append("--")
                elif not r.needle_present_in_context:
                    row_values.append("N/A")
                elif r.generated_text == "OOM":
                    row_values.append("OOM")
                elif r.generated_text.startswith("ERROR"):
                    row_values.append("ERR")
                elif r.passed:
                    row_values.append("PASS")
                else:
                    row_values.append("FAIL")

            print(fmt_row.format(str(actual_ctx), f"{depth}%", *row_values))

    print()

    # Summary stats
    total = sum(1 for r in results if r.needle_present_in_context)
    if total > 0:
        for cname in config_names:
            relevant = [r for r in results
                        if r.config_name == cname and r.needle_present_in_context]
            passed = sum(1 for r in relevant if r.passed)
            pct = 100.0 * passed / len(relevant) if relevant else 0.0
            print(f"  {cname:20s}: {passed}/{len(relevant)} passed ({pct:.0f}%)")
    print()


def _print_failures(results: List[TrialResult]):
    """Print details of failed trials for debugging."""
    failures = [r for r in results
                if not r.passed and r.needle_present_in_context
                and r.generated_text not in ("OOM",) and not r.generated_text.startswith("ERROR")]

    if not failures:
        print("All trials with needle present: PASSED (or OOM/error).\n")
        return

    print(f"--- Failed Trials ({len(failures)}) ---\n")
    for r in failures[:10]:  # limit output
        print(f"  ctx={r.context_len} depth={r.depth_pct}% [{r.config_name}]")
        snippet = r.generated_text[:150].replace("\n", " ")
        print(f"    Generated: {snippet}")
        print()


if __name__ == "__main__":
    main()
