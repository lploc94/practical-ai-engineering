# eval_gate.py: chặn merge khi candidate tệ hơn baseline có ý nghĩa thống kê
import json, sys
from collections import defaultdict
from eval_stats import paired_diff_ci       # hàm ở §1.4

MARGIN = 0.02        # chấp nhận tụt tối đa 2 điểm % trước khi chặn

def load(path: str) -> tuple[dict, dict]:
    runs, groups = defaultdict(list), {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            runs[r["case_id"]].append(1.0 if r["pass"] else 0.0)
            groups[r["case_id"]] = r["group"]
    return {k: sum(v) / len(v) for k, v in runs.items()}, groups

base, groups = load(sys.argv[1])
cand, _ = load(sys.argv[2])
missing, extra = base.keys() - cand.keys(), cand.keys() - base.keys()
if missing or extra:       # lần chạy dở dang hoặc dataset lệch: không so được, không được coi là pass
    print(f"case không khớp: thiếu {sorted(missing)[:10]}, thừa {sorted(extra)[:10]}")
    sys.exit(2)
failed = False
for g in sorted(set(groups.values())) + ["__all__"]:
    ids = {i for i, gi in groups.items() if g == "__all__" or gi == g}
    b = {i: base[i] for i in ids}
    c = {i: cand[i] for i in ids}
    diff, lo, hi = paired_diff_ci(c, b)
    print(f"{g:20s} n={len(b):4d} diff={diff:+.3f} CI95=[{lo:+.3f}, {hi:+.3f}]")
    if hi < -MARGIN:       # cả khoảng tin cậy nằm dưới -2 điểm: tệ đi chắc chắn
        failed = True
sys.exit(1 if failed else 0)
