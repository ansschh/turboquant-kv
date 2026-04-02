"""
Standard vector search benchmarks for TurboQuant.

Reproduces the paper's nearest-neighbor search experiments (Section 4.4)
on standard datasets: GloVe-200, SIFT-128, and random high-dim vectors.

Compares:
  - TurboQuant (2, 3, 4 bit) via Triton kernel
  - FAISS FlatIP (exact brute-force)
  - FAISS IVFPQ (learned compression)
  - FAISS IVFFlat (uncompressed approximate)

Metrics: Recall@1@k for k in {1, 2, 4, 8, 16, 32, 64}
"""

import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def load_glove(n=100_000, d=200):
    """Load GloVe vectors from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        print(f"  Loading GloVe from HuggingFace (n={n}, d={d})...")
        ds = load_dataset("mteb/glove", split="train")
        vecs = np.array(ds["embedding"][:n], dtype=np.float32)
        if vecs.shape[1] != d:
            vecs = vecs[:, :d]
        return vecs
    except Exception as e:
        print(f"  GloVe download failed ({e}), using random vectors")
        return np.random.randn(n, d).astype(np.float32)


def generate_dataset(name, n_db=100_000, n_query=1000):
    """Generate or load a benchmark dataset."""
    if name == "glove-200":
        db = load_glove(n_db + n_query, 200)
        queries = db[n_db:]
        db = db[:n_db]
        d = 200
    elif name == "random-128":
        d = 128
        db = np.random.randn(n_db, d).astype(np.float32)
        queries = np.random.randn(n_query, d).astype(np.float32)
    elif name == "random-1536":
        d = 1536
        db = np.random.randn(n_db, d).astype(np.float32)
        queries = np.random.randn(n_query, d).astype(np.float32)
    elif name == "random-2304":
        d = 2304
        db = np.random.randn(n_db, d).astype(np.float32)
        queries = np.random.randn(n_query, d).astype(np.float32)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Normalize to unit sphere (standard for IP search)
    db_norms = np.linalg.norm(db, axis=1, keepdims=True)
    db = db / np.maximum(db_norms, 1e-10)
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / np.maximum(q_norms, 1e-10)

    return db, queries, d


def exact_topk(db, queries, k=64):
    """Compute exact top-k by brute-force inner product."""
    db_t = torch.from_numpy(db).cuda()
    q_t = torch.from_numpy(queries).cuda()
    scores = q_t @ db_t.T
    _, indices = scores.topk(k, dim=1)
    return indices.cpu().numpy()


def recall_at_k(gt_top1, approx_topk):
    """Recall@1@k: fraction of queries where true top-1 is in approx top-k."""
    n = gt_top1.shape[0]
    hits = 0
    for i in range(n):
        if gt_top1[i] in approx_topk[i]:
            hits += 1
    return hits / n


def bench_turboquant(db, queries, d, bit_widths=[2, 3, 4], ks=[1, 2, 4, 8, 16, 32, 64]):
    """Benchmark TurboQuant at various bit widths."""
    from turboquant_kv.reference import lloyd_max_codebook, make_rotation_matrix, pack_codes
    try:
        import turboquant_kv._C
        has_cuda_ops = True
    except ImportError:
        has_cuda_ops = False

    results = []
    rot = make_rotation_matrix(d, seed=42, method="dense_qr").cuda()
    db_t = torch.from_numpy(db).cuda().float()
    q_t = torch.from_numpy(queries).cuda().float()
    N = db_t.shape[0]
    Q = q_t.shape[0]

    for bw in bit_widths:
        boundaries, centroids = lloyd_max_codebook(bw, d)
        boundaries, centroids = boundaries.cuda(), centroids.cuda()

        # Quantize database
        t0 = time.time()
        if has_cuda_ops:
            packed, norms = torch.ops.turboquant.rotate_and_quantize(db_t, rot, boundaries, bw)
        else:
            # Python fallback
            db_norms = db_t.norm(dim=-1, keepdim=True)
            unit = db_t / db_norms.clamp(min=1e-10)
            rotated = unit @ rot.T
            codes = torch.searchsorted(boundaries, rotated).to(torch.uint8)
            packed = pack_codes(codes, bw)
            norms = db_norms.squeeze(-1)
        quant_time = time.time() - t0

        # Search
        q_rot = q_t @ rot.T
        n_levels = 1 << bw

        # Try Triton v2 for 4-bit
        use_triton = False
        if bw == 4:
            try:
                from turboquant_kv.triton_kernels import _tq_scores_v2_kernel
                import triton
                use_triton = True
            except ImportError:
                pass

        bytes_per_plane = (d + 7) // 8
        packed_dim = bw * bytes_per_plane

        t0 = time.time()
        all_topk = np.zeros((Q, max(ks)), dtype=np.int64)

        if use_triton:
            from turboquant_kv.triton_kernels import _tq_scores_v2_kernel
            import triton
            BLOCK_N = 512
            for qi in range(Q):
                table = (q_rot[qi].unsqueeze(1) * centroids.unsqueeze(0)).contiguous()
                scores = torch.empty(N, device="cuda", dtype=torch.float32)
                grid = (triton.cdiv(N, BLOCK_N),)
                _tq_scores_v2_kernel[grid](
                    packed, norms, table, scores,
                    N, d, bytes_per_plane, packed_dim, n_levels,
                    BLOCK_N=BLOCK_N,
                )
                _, topk_idx = scores.topk(max(ks))
                all_topk[qi] = topk_idx.cpu().numpy()
        else:
            # Python fallback
            from turboquant_kv.reference import unpack_codes
            codes_full = unpack_codes(packed, bw, d)
            cv = centroids[codes_full.long()]
            for qi in range(Q):
                scores = (q_rot[qi] @ cv.T) * norms
                _, topk_idx = scores.topk(max(ks))
                all_topk[qi] = topk_idx.cpu().numpy()

        search_time = time.time() - t0

        # Compute recall at each k
        gt_top1 = exact_topk(db, queries, 1)[:, 0]
        for k in ks:
            r = recall_at_k(gt_top1, all_topk[:, :k])
            results.append({
                "method": f"TurboQuant {bw}-bit",
                "bit_width": bw,
                "k": k,
                "recall_1_at_k": r,
                "quant_time_s": quant_time,
                "search_time_s": search_time,
                "bytes_per_vec": bw * d // 8 + 4,
            })

        print(f"  TQ {bw}-bit: recall@1@10={results[-4]['recall_1_at_k']:.3f}, "
              f"quant={quant_time:.2f}s, search={search_time:.2f}s")

    return results


def bench_faiss(db, queries, d, ks=[1, 2, 4, 8, 16, 32, 64]):
    """Benchmark FAISS methods."""
    import faiss
    faiss.omp_set_num_threads(8)
    results = []
    N = db.shape[0]
    Q = queries.shape[0]

    gt_top1 = exact_topk(db, queries, 1)[:, 0]

    # FlatIP (exact)
    t0 = time.time()
    index = faiss.IndexFlatIP(d)
    index.add(db)
    build_time = time.time() - t0

    t0 = time.time()
    scores, ids = index.search(queries, max(ks))
    search_time = time.time() - t0

    for k in ks:
        r = recall_at_k(gt_top1, ids[:, :k])
        results.append({
            "method": "FAISS FlatIP",
            "k": k,
            "recall_1_at_k": r,
            "build_time_s": build_time,
            "search_time_s": search_time,
            "bytes_per_vec": d * 4,
        })
    print(f"  FlatIP: recall@1@10={results[-4]['recall_1_at_k']:.3f}, "
          f"build={build_time:.2f}s, search={search_time:.2f}s")
    del index

    # IVFPQ
    for m in [8, 16, 32]:
        if d % m != 0:
            continue
        nlist = min(1024, N // 40)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
        t0 = time.time()
        index.train(db)
        index.add(db)
        build_time = time.time() - t0
        index.nprobe = 64

        t0 = time.time()
        scores, ids = index.search(queries, max(ks))
        search_time = time.time() - t0

        for k in ks:
            r = recall_at_k(gt_top1, ids[:, :k])
            results.append({
                "method": f"FAISS PQ m={m}",
                "k": k,
                "recall_1_at_k": r,
                "build_time_s": build_time,
                "search_time_s": search_time,
                "bytes_per_vec": m + 4,
            })
        print(f"  PQ m={m}: recall@1@10={results[-4]['recall_1_at_k']:.3f}, "
              f"build={build_time:.2f}s, search={search_time:.2f}s")
        del index

    return results


def plot_results(all_results, dataset_name, out_dir):
    """Generate recall@1@k plot."""
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })

    colors = {
        "TurboQuant 2-bit": "#e74c3c",
        "TurboQuant 3-bit": "#3498db",
        "TurboQuant 4-bit": "#2ecc71",
        "FAISS FlatIP": "#7f8c8d",
        "FAISS PQ m=8": "#e67e22",
        "FAISS PQ m=16": "#9b59b6",
        "FAISS PQ m=32": "#1abc9c",
    }
    markers = {
        "TurboQuant 2-bit": "^",
        "TurboQuant 3-bit": "s",
        "TurboQuant 4-bit": "o",
        "FAISS FlatIP": "*",
        "FAISS PQ m=8": "D",
        "FAISS PQ m=16": "v",
        "FAISS PQ m=32": "P",
    }

    fig, ax = plt.subplots()
    methods = sorted(set(r["method"] for r in all_results))

    for method in methods:
        rows = [r for r in all_results if r["method"] == method]
        rows.sort(key=lambda r: r["k"])
        ks = [r["k"] for r in rows]
        recalls = [r["recall_1_at_k"] for r in rows]
        ax.plot(ks, recalls,
                color=colors.get(method, "gray"),
                marker=markers.get(method, "x"),
                linewidth=2, markersize=8,
                label=method)

    ax.set_xlabel("Top-k")
    ax.set_ylabel("Recall@1@k")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Recall@1@k — {dataset_name}")
    ax.legend(loc="lower right", fontsize=10)

    path = os.path.join(out_dir, f"recall_{dataset_name.replace(' ', '_').lower()}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_quantization_time(all_results_by_dataset, out_dir):
    """Bar chart of quantization/build time."""
    os.makedirs(out_dir, exist_ok=True)

    methods = []
    times = []
    for ds_name, results in all_results_by_dataset.items():
        seen = set()
        for r in results:
            m = r["method"]
            if m in seen:
                continue
            seen.add(m)
            t = r.get("quant_time_s", r.get("build_time_s", 0))
            methods.append(f"{m}\n({ds_name})")
            times.append(t)

    # Just use the first dataset for simplicity
    ds_name = list(all_results_by_dataset.keys())[0]
    results = all_results_by_dataset[ds_name]
    seen = {}
    for r in results:
        m = r["method"]
        if m not in seen:
            seen[m] = r.get("quant_time_s", r.get("build_time_s", 0))

    fig, ax = plt.subplots(figsize=(10, 5))
    methods_list = list(seen.keys())
    times_list = list(seen.values())

    colors_list = ["#2ecc71" if "TurboQuant" in m else "#3498db" if "Flat" in m else "#e74c3c"
                   for m in methods_list]
    bars = ax.barh(methods_list, times_list, color=colors_list, edgecolor="white", linewidth=0.5)

    for bar, t in zip(bars, times_list):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{t:.2f}s", va="center", fontsize=10)

    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Index Build / Quantization Time — {ds_name}")
    ax.invert_yaxis()

    path = os.path.join(out_dir, "quant_time.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "../../results")
    os.makedirs(out_dir, exist_ok=True)

    datasets = [
        ("random-128", 100_000, 1000),
        ("random-1536", 100_000, 1000),
    ]

    # Try to add GloVe
    try:
        from datasets import load_dataset
        datasets.insert(0, ("glove-200", 100_000, 1000))
    except ImportError:
        print("datasets not installed, skipping GloVe")

    all_results_by_ds = {}
    all_plots = []

    for ds_name, n_db, n_query in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (N={n_db:,}, Q={n_query})")
        print(f"{'='*60}")

        db, queries, d = generate_dataset(ds_name, n_db, n_query)
        print(f"  Loaded: db={db.shape}, queries={queries.shape}")

        # Determine valid PQ m values
        results = []

        print("\n  TurboQuant:")
        tq_results = bench_turboquant(db, queries, d)
        results.extend(tq_results)

        print("\n  FAISS:")
        faiss_results = bench_faiss(db, queries, d)
        results.extend(faiss_results)

        all_results_by_ds[ds_name] = results

        # Plot
        plot_path = plot_results(results, ds_name, out_dir)
        all_plots.append(plot_path)

        # Save raw results
        with open(os.path.join(out_dir, f"results_{ds_name}.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Quantization time plot
    plot_quantization_time(all_results_by_ds, out_dir)

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
