from collections import defaultdict
import numpy as np

def paired_diff_ci(cand: dict, base: dict, clusters: dict | None = None,
                   n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """cand, base: {case_id: điểm 0..1, trung bình các trial} của hai phiên bản.
    clusters: {case_id: nhóm} khi nhiều ca chung một tài liệu hoặc hội thoại.
    Trả về (hiệu số trung bình, cận dưới, cận trên) của khoảng tin cậy 95%."""
    ids = sorted(cand.keys() & base.keys())
    groups = defaultdict(list)
    for i in ids:
        groups[clusters[i] if clusters else i].append(cand[i] - base[i])
    sums = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))   # resample theo nhóm
    boot = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(sums.sum() / sizes.sum()), float(lo), float(hi)
