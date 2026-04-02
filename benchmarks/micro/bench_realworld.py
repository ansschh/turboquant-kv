"""
Real-world benchmarks: GloVe-200 (400K words) and SIFT-1M (1M image descriptors).

Compares TurboQuant (2,3,4-bit) vs FAISS FlatIP vs FAISS IVFPQ.
Metric: Recall@1@k for k in {1, 2, 4, 8, 16, 32, 64}
"""

import os, sys, time, json
import numpy as np
import torch
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

OUT_DIR = os.path.join(os.path.dirname(__file__), "../../results")
os.makedirs(OUT_DIR, exist_ok=True)
KS = [1, 2, 4, 8, 16, 32, 64]


def compute_gt(train, test, k=100):
    d = train.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(train)
    _, ids = index.search(test, k)
    return ids


def recall_1_at_k(gt1, topk):
    return sum(1 for i in range(len(gt1)) if gt1[i] in topk[i]) / len(gt1)


def bench_tq(train, test, d, gt1, bits=[2, 3, 4]):
    from turboquant_kv.reference import lloyd_max_codebook, make_rotation_matrix
    try:
        import turboquant_kv._C
        has_ops = True
    except ImportError:
        has_ops = False

    results = []
    rot = make_rotation_matrix(d, seed=42, method="dense_qr").cuda()
    db = torch.from_numpy(train).cuda().float()
    qt = torch.from_numpy(test).cuda().float()
    N, Q = db.shape[0], qt.shape[0]

    for bw in bits:
        boundaries, centroids = lloyd_max_codebook(bw, d)
        boundaries, centroids = boundaries.cuda(), centroids.cuda()

        t0 = time.time()
        if has_ops:
            packed, norms = torch.ops.turboquant.rotate_and_quantize(db, rot, boundaries, bw)
        else:
            from turboquant_kv.reference import pack_codes
            db_n = db.norm(dim=-1, keepdim=True)
            unit = db / db_n.clamp(min=1e-10)
            rotated = unit @ rot.T
            codes = torch.searchsorted(boundaries, rotated).to(torch.uint8)
            packed = pack_codes(codes, bw)
            norms = db_n.squeeze(-1)
        qt_s = time.time() - t0

        q_rot = qt @ rot.T
        nl = 1 << bw
        bpp = (d + 7) // 8
        pd = bw * bpp

        use_triton = bw == 4
        if use_triton:
            try:
                from turboquant_kv.triton_kernels import _tq_scores_v2_kernel
                import triton
            except ImportError:
                use_triton = False

        t0 = time.time()
        mk = max(KS)
        topk_all = np.zeros((Q, mk), dtype=np.int64)

        if use_triton:
            import triton
            BN = 512
            for qi in range(Q):
                tab = (q_rot[qi].unsqueeze(1) * centroids.unsqueeze(0)).contiguous()
                sc = torch.empty(N, device="cuda", dtype=torch.float32)
                _tq_scores_v2_kernel[(triton.cdiv(N, BN),)](
                    packed, norms, tab, sc, N, d, bpp, pd, nl, BLOCK_N=BN)
                _, ti = sc.topk(mk)
                topk_all[qi] = ti.cpu().numpy()
        else:
            from turboquant_kv.reference import unpack_codes
            codes = unpack_codes(packed, bw, d)
            cv = centroids[codes.long()]
            for qi in range(Q):
                sc = (q_rot[qi] @ cv.T) * norms
                _, ti = sc.topk(mk)
                topk_all[qi] = ti.cpu().numpy()

        st = time.time() - t0

        for k in KS:
            results.append({
                "method": f"TurboQuant {bw}-bit", "k": k,
                "recall_1_at_k": round(recall_1_at_k(gt1, topk_all[:, :k]), 4),
                "quant_time_s": round(qt_s, 3), "search_time_s": round(st, 3),
                "bytes_per_vec": bw * d // 8 + 4,
            })
        r10 = recall_1_at_k(gt1, topk_all[:, :10])
        print(f"    TQ {bw}-bit: R@1@10={r10:.3f}  quant={qt_s:.2f}s  search={st:.1f}s")

    return results


def bench_faiss(train, test, d, gt1):
    results = []
    N = train.shape[0]
    mk = max(KS)
    faiss.omp_set_num_threads(8)

    # FlatIP
    t0 = time.time()
    idx = faiss.IndexFlatIP(d)
    idx.add(train)
    bt = time.time() - t0
    t0 = time.time()
    _, ids = idx.search(test, mk)
    st = time.time() - t0
    for k in KS:
        results.append({"method": "FAISS FlatIP", "k": k,
                        "recall_1_at_k": round(recall_1_at_k(gt1, ids[:, :k]), 4),
                        "build_time_s": round(bt, 3), "search_time_s": round(st, 3),
                        "bytes_per_vec": d * 4})
    print(f"    FlatIP: R@1@10={recall_1_at_k(gt1, ids[:,:10]):.3f}  build={bt:.2f}s")
    del idx

    # IVFPQ
    for m in [8, 16, 32]:
        if d % m != 0:
            continue
        nl = min(1024, N // 40)
        q = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFPQ(q, d, nl, m, 8, faiss.METRIC_INNER_PRODUCT)
        t0 = time.time()
        idx.train(train)
        idx.add(train)
        bt = time.time() - t0
        idx.nprobe = 64
        t0 = time.time()
        _, ids = idx.search(test, mk)
        st = time.time() - t0
        for k in KS:
            results.append({"method": f"FAISS PQ m={m}", "k": k,
                            "recall_1_at_k": round(recall_1_at_k(gt1, ids[:, :k]), 4),
                            "build_time_s": round(bt, 3), "search_time_s": round(st, 3),
                            "bytes_per_vec": m + 4})
        print(f"    PQ m={m}: R@1@10={recall_1_at_k(gt1, ids[:,:10]):.3f}  build={bt:.1f}s")
        del idx

    return results


def plot_recall(results, name, out_dir):
    plt.rcParams.update({
        "figure.figsize": (10, 6.5), "font.size": 13, "font.family": "sans-serif",
        "axes.grid": True, "grid.alpha": 0.2,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    })
    colors = {
        "TurboQuant 2-bit": "#e74c3c", "TurboQuant 3-bit": "#3498db",
        "TurboQuant 4-bit": "#2ecc71", "FAISS FlatIP": "#95a5a6",
        "FAISS PQ m=8": "#e67e22", "FAISS PQ m=16": "#9b59b6",
        "FAISS PQ m=32": "#1abc9c",
    }
    markers = {
        "TurboQuant 2-bit": "^", "TurboQuant 3-bit": "s",
        "TurboQuant 4-bit": "o", "FAISS FlatIP": "*",
        "FAISS PQ m=8": "D", "FAISS PQ m=16": "v", "FAISS PQ m=32": "P",
    }

    fig, ax = plt.subplots()
    methods = sorted(set(r["method"] for r in results),
                     key=lambda m: -max(r["recall_1_at_k"] for r in results if r["method"] == m))
    for method in methods:
        rows = sorted([r for r in results if r["method"] == method], key=lambda r: r["k"])
        ax.plot([r["k"] for r in rows], [r["recall_1_at_k"] for r in rows],
                color=colors.get(method, "gray"), marker=markers.get(method, "x"),
                linewidth=2.5, markersize=9, label=method, alpha=0.9)

    ax.set_xlabel("Top-k", fontsize=14)
    ax.set_ylabel("Recall@1@k", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Recall@1@k — {name}", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)

    path = os.path.join(out_dir, f"recall_{name.lower().replace(' ', '_').replace('-', '_')}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Plot: {path}")


def main():
    datasets = []

    # GloVe-200 (real word embeddings, 400K, d=200)
    glove_path = "/tmp/glove_200d.npy"
    if os.path.exists(glove_path):
        vecs = np.load(glove_path)
        n_q = min(10000, len(vecs) // 10)
        datasets.append(("GloVe-200", vecs[:-n_q], vecs[-n_q:], 200))
    else:
        print(f"GloVe not found at {glove_path}")

    # SIFT-1M (real image descriptors, 1M, d=128)
    sift_base = "/tmp/sift_base.npy"
    sift_query = "/tmp/sift_query.npy"
    if os.path.exists(sift_base) and os.path.exists(sift_query):
        datasets.append(("SIFT-1M", np.load(sift_base), np.load(sift_query), 128))
    else:
        print(f"SIFT not found at {sift_base}")

    if not datasets:
        print("No datasets found. Upload glove_200d.npy and sift_base.npy to /tmp/")
        return

    for name, train, test, d in datasets:
        N, Q = train.shape[0], test.shape[0]
        print(f"\n{'='*60}")
        print(f"{name}: N={N:,}, Q={Q:,}, d={d}")
        print(f"{'='*60}")

        # Normalize for IP search
        train = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-10)
        test = test / np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-10)

        print("  Ground truth...")
        gt = compute_gt(train, test, k=1)
        gt1 = gt[:, 0]

        results = []
        print("  TurboQuant:")
        results.extend(bench_tq(train, test, d, gt1))
        print("  FAISS:")
        results.extend(bench_faiss(train, test, d, gt1))

        plot_recall(results, name, OUT_DIR)
        with open(os.path.join(OUT_DIR, f"results_{name.lower().replace(' ','_')}.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
