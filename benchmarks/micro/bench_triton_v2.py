"""Benchmark optimized Triton v2 kernel with precomputed lookup table."""
import torch, time, triton, triton.language as tl
import sys
sys.path.insert(0, "/workspace/turboquant_kv")


@triton.jit
def _tq_scores_v2(
    packed_ptr, norms_ptr, table_ptr, scores_ptr,
    seq_len, head_dim,
    bytes_per_plane: tl.constexpr,
    packed_dim: tl.constexpr,
    n_levels: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    vec_start = pid * BLOCK_N
    offs = vec_start + tl.arange(0, BLOCK_N)
    mask = offs < seq_len

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for byte_idx in range(bytes_per_plane):
        b0 = tl.load(packed_ptr + offs * packed_dim + 0 * bytes_per_plane + byte_idx,
                      mask=mask, other=0).to(tl.int32)
        b1 = tl.load(packed_ptr + offs * packed_dim + 1 * bytes_per_plane + byte_idx,
                      mask=mask, other=0).to(tl.int32)
        b2 = tl.load(packed_ptr + offs * packed_dim + 2 * bytes_per_plane + byte_idx,
                      mask=mask, other=0).to(tl.int32)
        b3 = tl.load(packed_ptr + offs * packed_dim + 3 * bytes_per_plane + byte_idx,
                      mask=mask, other=0).to(tl.int32)

        base_dim = byte_idx * 8
        for bp_inv in tl.static_range(8):
            d_idx = base_dim + bp_inv
            if d_idx < head_dim:
                bp = 7 - bp_inv
                code = ((b0 >> bp) & 1) | \
                       (((b1 >> bp) & 1) << 1) | \
                       (((b2 >> bp) & 1) << 2) | \
                       (((b3 >> bp) & 1) << 3)
                val = tl.load(table_ptr + d_idx * n_levels + code, mask=mask)
                acc += val

    norms = tl.load(norms_ptr + offs, mask=mask, other=0.0)
    acc = acc * norms
    tl.store(scores_ptr + offs, acc, mask=mask)


import turboquant_kv._C
from turboquant_kv.reference import lloyd_max_codebook, make_rotation_matrix, unpack_codes

d = 128
N = 10_000_000
b = 4

rot = make_rotation_matrix(d, seed=42, method="dense_qr").cuda()
boundaries, centroids = lloyd_max_codebook(b, d)
boundaries, centroids = boundaries.cuda(), centroids.cuda()

print("Building index...")
vecs = torch.randn(N, d, device="cuda")
packed, norms = torch.ops.turboquant.rotate_and_quantize(vecs, rot, boundaries, b)
query = torch.randn(1, d, device="cuda")
q_rot = (query @ rot.T).squeeze(0).float()

n_levels = 16
table = (q_rot.unsqueeze(1) * centroids.unsqueeze(0)).contiguous()
bytes_per_plane = d // 8
packed_dim = b * bytes_per_plane
scores = torch.empty(N, device="cuda", dtype=torch.float32)

# Warmup
for BN in [512, 1024, 2048, 4096]:
    grid = ((N + BN - 1) // BN,)
    _tq_scores_v2[grid](
        packed, norms, table, scores,
        N, d, bytes_per_plane, packed_dim, n_levels, BLOCK_N=BN,
    )
    torch.cuda.synchronize()

# Benchmark
print(f"\nN={N:,}, d={d}, b={b}")
for BN in [512, 1024, 2048, 4096]:
    grid = ((N + BN - 1) // BN,)
    torch.cuda.synchronize()
    t0 = time.time()
    reps = 50
    for _ in range(reps):
        _tq_scores_v2[grid](
            packed, norms, table, scores,
            N, d, bytes_per_plane, packed_dim, n_levels, BLOCK_N=BN,
        )
    torch.cuda.synchronize()
    ms = (time.time() - t0) / reps * 1000
    print(f"  BLOCK_N={BN:5d}: {ms:.2f} ms")

# Correctness
codes = unpack_codes(packed[:100], b, d)
cv = centroids[codes.long()]
manual = (q_rot @ cv.T) * norms[:100]
diff = (scores[:100] - manual).abs().max()
print(f"\nCorrectness: max diff = {diff:.6f}")

# cuBLAS baseline
vf = vecs.float()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    query @ vf.T
torch.cuda.synchronize()
flat_ms = (time.time() - t0) / 50 * 1000
print(f"cuBLAS fp32:   {flat_ms:.2f} ms")
print(f"Theoretical:   0.20 ms (bandwidth-limited)")

best_ms = min(ms for BN in [512, 1024, 2048, 4096]
              for ms in [0])  # placeholder
print(f"\nBest Triton / cuBLAS ratio will show above")
