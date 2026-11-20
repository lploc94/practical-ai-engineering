"""Sinh paged-kv.svg: cấp KV cache liền theo độ dài tối đa so với cấp theo block (§1.2).

Kịch bản giả định để minh họa đoạn PagedAttention ở §1.2 [1]:
- Vùng KV cache có 24 block; độ dài tối đa (max_model_len) ứng với 8 block.
- Đang chạy A, B, C, hiện dùng 3, 5, 2 block. D, E đang chờ, cần 3 và 2 block.
- Cấp liền: mỗi request giữ sẵn 8 block liền nhau, 3 request chiếm hết 24 block
  dù chỉ dùng 10; D, E phải chờ.
- Cấp theo block (PagedAttention): block cấp khi request cần thêm, không cần liền
  nhau (thứ tự xen kẽ vì các request lớn lên cùng lúc); A, B, C chiếm 10 block,
  D, E vào được, còn 9 block trống.
Chỉ minh họa một ý: số request đồng thời phụ thuộc tổng token đang có, không phải
max_model_len × số request. Hình tĩnh.

Chạy: python3 make_paged_kv.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

POOL, MAXB = 24, 8
USED = {"A": 3, "B": 5, "C": 2}
WAIT = {"D": 3, "E": 2}

# Cấp liền: mỗi request giữ MAXB block
contig: list[tuple[str, bool]] = []          # (request, đang dùng?)
for r, n in USED.items():
    contig += [(r, True)] * n + [(r, False)] * (MAXB - n)
assert len(contig) == POOL                    # hết chỗ cho D, E
assert POOL // MAXB == len(USED)

# Cấp theo block: thứ tự cấp xen kẽ khi các request lớn lên
paged = list("AABBBCABBC") + list("DDDEE")
for r, n in {**USED, **WAIT}.items():
    assert paged.count(r) == n, r
FREE = POOL - len(paged)
assert sum(USED.values()) == 10 and FREE == 9

CW, CH, GAP = 30, 34, 4
X0 = 170
W = X0 + POOL * (CW + GAP) - GAP + 36
Y1, Y2 = 120, 240
H = Y2 + CH + 110


def cx(i: int) -> float:
    return X0 + i * (CW + GAP)


def cell(i: int, y: float, cls: str, label: str = "") -> str:
    out = f'<rect class="{cls}" x="{cx(i)}" y="{y}" width="{CW}" height="{CH}" rx="4"/>'
    if label:
        out += f'<text class="lbl" style="font-size:13px" x="{cx(i) + CW / 2}" y="{y + CH / 2 + 5}">{label}</text>'
    return out


body: list[str] = []
body.append('<text class="title" x="20" y="38">KV cache: cấp liền theo độ dài tối đa hay theo block</text>')
body.append(svgkit.legend(20, 72, [("blue", "A, B, C đang dùng"), ("tok", "giữ sẵn, chưa dùng"),
                                    ("orange", "D, E vào thêm"), ("slot", "block trống")]))

# Hàng 1: cấp liền
body.append(f'<text class="row" x="20" y="{Y1 + 15}">Cấp liền</text>')
body.append(f'<text class="sub" x="20" y="{Y1 + 32}">8 block mỗi request</text>')
for i, (r, used) in enumerate(contig):
    body.append(cell(i, Y1, "blue" if used else "tok", r if used else ""))
for k, r in enumerate(USED):
    xa, xb = cx(k * MAXB), cx(k * MAXB + MAXB - 1) + CW
    body.append(f'<path class="lmuted" style="stroke:var(--muted)" d="M{xa},{Y1 - 6} v-6 H{xb} v6"/>'
                f'<text class="sub mid" x="{(xa + xb) / 2}" y="{Y1 - 18}">{r} giữ {MAXB} block, dùng {USED[r]}</text>')
body.append(f'<text class="cap" x="{X0}" y="{Y1 + CH + 26}">Dùng {sum(USED.values())} / {POOL} block '
            f'nhưng đã hết chỗ: D, E phải chờ.</text>')

# Hàng 2: theo block
body.append(f'<text class="row" x="20" y="{Y2 + 15}">Theo block</text>')
body.append(f'<text class="sub" x="20" y="{Y2 + 32}">PagedAttention</text>')
for i in range(POOL):
    if i < len(paged):
        r = paged[i]
        body.append(cell(i, Y2, "orange" if r in WAIT else "blue", r))
    else:
        body.append(cell(i, Y2, "slot"))
body.append(f'<text class="cap" x="{X0}" y="{Y2 + CH + 26}">Block cấp khi cần, không phải liền nhau. '
            f'D, E vào được, còn {FREE} block trống.</text>')
body.append(f'<text class="cap2" x="20" y="{H - 16}">Số request chạy cùng lúc phụ thuộc tổng token đang có, '
            'không phải độ dài tối đa × số request. Số block giả định.</text>')

svgkit.write(__file__, "paged-kv.svg", svgkit.svg(
    W, H, "Cấp KV cache liền theo độ dài tối đa: ba request giữ hết 24 block dù chỉ dùng 10, request "
    "khác phải chờ. Cấp theo block như PagedAttention: chỉ chiếm phần đang dùng, thêm được hai request "
    "và còn 9 block trống", body))
