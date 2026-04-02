"""Comprehensive quality sweep on Llama-3.2-1B-Instruct."""
import torch, math, time, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant_kv.hf_integration import TurboQuantCache
from datasets import load_dataset

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.2-1B-Instruct")
OUT = os.path.join(os.path.dirname(__file__), "../results")
os.makedirs(OUT, exist_ok=True)

print(f"Loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")

ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
text = " ".join([r["text"] for r in ds if r["text"].strip()][:500])

configs = [
    (4, 4, "K4/V4"),
    (4, 3, "K4/V3"),
    (4, 2, "K4/V2"),
    (3, 3, "K3/V3"),
    (3, 2, "K3/V2"),
    (2, 2, "K2/V2"),
]

results = []

for ctx_len in [128, 256, 512, 1024]:
    ids = tok.encode(text, return_tensors="pt", max_length=ctx_len, truncation=True).cuda()
    T = ids.shape[1]

    with torch.no_grad():
        out_base = model(ids)
    logit_base = out_base.logits[0, -1].float()
    top5_base = set(logit_base.topk(5).indices.tolist())
    top10_base = set(logit_base.topk(10).indices.tolist())

    for bk, bv, label in configs:
        cache = TurboQuantCache(key_bits=bk, value_bits=bv)
        t0 = time.time()
        with torch.no_grad():
            for i in range(T):
                out = model(ids[:, i:i+1], past_key_values=cache, use_cache=True)
        elapsed = time.time() - t0
        logit_tq = out.logits[0, -1].float()

        cos = torch.nn.functional.cosine_similarity(
            logit_base.unsqueeze(0), logit_tq.unsqueeze(0), dim=-1
        ).item()
        top1_match = logit_base.argmax().item() == logit_tq.argmax().item()
        top5_tq = set(logit_tq.topk(5).indices.tolist())
        top10_tq = set(logit_tq.topk(10).indices.tolist())
        top5_overlap = len(top5_base & top5_tq) / 5
        top10_overlap = len(top10_base & top10_tq) / 10

        kv_bytes = cache.memory_bytes()

        row = {
            "ctx": T, "config": label, "key_bits": bk, "value_bits": bv,
            "cosine_sim": round(cos, 5),
            "top1_match": top1_match,
            "top5_overlap": round(top5_overlap, 2),
            "top10_overlap": round(top10_overlap, 2),
            "kv_mb": round(kv_bytes / 1e6, 1),
            "time_s": round(elapsed, 1),
        }
        results.append(row)
        print(f"  ctx={T:4d}  {label:6s}  cos={cos:.4f}  top1={'Y' if top1_match else 'N'}  "
              f"top5={top5_overlap:.0%}  top10={top10_overlap:.0%}  kv={kv_bytes/1e6:.1f}MB  {elapsed:.1f}s")

# Summary table
print()
print("=" * 90)
print("LLAMA-3.2-1B-INSTRUCT QUALITY SWEEP")
print("=" * 90)
header = f"{'ctx':>5} {'config':>7} {'cosine':>8} {'top1':>5} {'top5':>6} {'top10':>6} {'KV MB':>7}"
print(header)
print("-" * 90)
for r in results:
    print(f"{r['ctx']:>5} {r['config']:>7} {r['cosine_sim']:>8.4f} "
          f"{'Y' if r['top1_match'] else 'N':>5} {r['top5_overlap']:>6.0%} "
          f"{r['top10_overlap']:>6.0%} {r['kv_mb']:>7.1f}")

with open(os.path.join(OUT, "llama_quality_sweep.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}/llama_quality_sweep.json")
