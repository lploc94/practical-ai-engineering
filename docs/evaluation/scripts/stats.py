import math
from collections import defaultdict
import numpy as np

def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy 95% cho tỉ lệ pass; ổn định hơn p ± 1,96·SE khi n nhỏ hoặc p gần 0, 1."""
    if n == 0:
        return (0.0, 1.0)
    p = passes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

def paired_diff_ci(a: dict, b: dict, clusters: dict | None = None,
                   n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """a, b: {case_id: điểm 0..1 (trung bình các trial)} của candidate và baseline.
    clusters: {case_id: nhóm} khi nhiều ca chung một tài liệu/hội thoại; resample theo nhóm."""
    ids = sorted(a.keys() & b.keys())
    groups = defaultdict(list)
    for i in ids:
        groups[clusters[i] if clusters else i].append(a[i] - b[i])
    sums = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return sums.sum() / sizes.sum(), lo, hi

def judge_rates(human: list[int], judge: list[int]) -> tuple[float, float]:
    """1 = Pass, 0 = Fail. TPR: tỉ lệ ca tốt judge cho Pass; TNR: tỉ lệ ca hỏng judge cho Fail."""
    tp = sum(h == 1 and j == 1 for h, j in zip(human, judge))
    tn = sum(h == 0 and j == 0 for h, j in zip(human, judge))
    pos = sum(human)
    neg = len(human) - pos
    return tp / pos, tn / neg

def corrected_pass_rate(p_obs: float, tpr: float, tnr: float) -> float:
    """Hiệu chỉnh Rogan-Gladen: ước lượng tỉ lệ pass thật từ tỉ lệ pass judge báo."""
    youden = tpr + tnr - 1
    if youden <= 0:
        raise ValueError("judge không tốt hơn đoán ngẫu nhiên")
    return min(1.0, max(0.0, (p_obs + tnr - 1) / youden))

if __name__ == "__main__":
    print(wilson_interval(80, 100), wilson_interval(9, 10), wilson_interval(0, 20))
    rng = np.random.default_rng(1)
    base = {i: float(rng.random() < 0.80) for i in range(300)}
    cand = {i: (base[i] if rng.random() < 0.9 else float(rng.random() < 0.85)) for i in range(300)}
    print(paired_diff_ci(cand, base))
    print(paired_diff_ci(cand, base, clusters={i: i // 5 for i in range(300)}))
    print(judge_rates([1,1,1,0,0], [1,1,0,0,1]))
    print(round(corrected_pass_rate(0.85, 0.92, 0.90), 4))
