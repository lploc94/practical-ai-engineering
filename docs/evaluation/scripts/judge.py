def judge_rates(human: list[int], judge: list[int]) -> tuple[float, float]:
    """1 = Pass, 0 = Fail, trên tập test. Trả về (TPR, TNR)."""
    if len(human) != len(judge):
        raise ValueError("số nhãn người và nhãn judge khác nhau")
    if not set(human) | set(judge) <= {0, 1}:
        raise ValueError("nhãn chỉ được là 0 hoặc 1")
    pos = sum(human)
    if pos == 0 or pos == len(human):
        raise ValueError("tập test cần có cả ca Pass và ca Fail")
    tp = sum(h == 1 and j == 1 for h, j in zip(human, judge))
    tn = sum(h == 0 and j == 0 for h, j in zip(human, judge))
    return tp / pos, tn / (len(human) - pos)

def corrected_pass_rate(p_obs: float, tpr: float, tnr: float) -> float:
    """Ước lượng pass rate thật từ pass rate judge báo (p_obs)."""
    if not all(0.0 <= v <= 1.0 for v in (p_obs, tpr, tnr)):   # cũng loại NaN
        raise ValueError("p_obs, tpr, tnr phải là xác suất trong [0, 1]")
    youden = tpr + tnr - 1
    if youden <= 0:
        raise ValueError("judge không tốt hơn đoán ngẫu nhiên")
    return min(1.0, max(0.0, (p_obs + tnr - 1) / youden))

corrected_pass_rate(0.85, tpr=0.92, tnr=0.90)   # ≈ 0,915
