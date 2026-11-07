from dataclasses import dataclass


@dataclass
class Pick:
    threshold: float         # chặn khi score >= threshold
    recall: float            # tỉ lệ mẫu vi phạm bị chặn
    fpr: float               # tỉ lệ mẫu an toàn bị chặn nhầm
    fpr_upper: float         # cận trên khoảng tin cậy 95% (Wilson) của fpr: fpr thật có thể cao tới đây


def pick_threshold(scores: list[float], labels: list[bool], max_fpr: float) -> Pick | None:
    """Ngưỡng bắt được nhiều vi phạm nhất mà tỉ lệ chặn nhầm trên mẫu an toàn không vượt
    `max_fpr`. `labels[i]` là True nếu mẫu i vi phạm. Trả None nếu không ngưỡng nào đạt."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        raise ValueError("cần cả mẫu vi phạm và mẫu an toàn")
    best = None
    for t in sorted(set(scores), reverse=True):          # ngưỡng giảm dần: recall tăng, fpr tăng
        tp = sum(1 for s, y in zip(scores, labels) if s >= t and y)
        fp = sum(1 for s, y in zip(scores, labels) if s >= t and not y)
        if fp / neg > max_fpr:
            break
        best = Pick(t, tp / pos, fp / neg, _wilson_upper(fp, neg))
    return best


def _wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return min(1.0, (centre + margin) / (1 + z * z / n))
