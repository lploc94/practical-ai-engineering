"""Sinh continuous-batching.svg: SVG động, static batching so với continuous batching (§1.3).

Kịch bản giả định, chỉ để minh họa cơ chế mà §1.3 mô tả theo Orca [45]:
- 3 slot (số request chạy cùng lúc trong một batch).
- Request A, B, C có sẵn lúc đầu, cần lần lượt 3, 6, 2 bước decode.
- Request D, E đang chờ trong hàng đợi, cần 3 và 2 bước.
- Static batching: batch đầu chạy tới khi request dài nhất (B) xong ở bước 6, rồi
  mới nhận D, E; D xong ở bước 9. Slot của A và C ngồi không từ khi chúng xong.
- Continuous batching (xếp lịch theo từng bước): C xong ở bước 2 thì D vào ở bước 3,
  A xong ở bước 3 thì E vào ở bước 4; mọi request xong ở bước 6.
Mỗi ô là một bước decode của một slot (bước đầu của request gồm cả prefill).
Hai hàng chạy trên cùng đồng hồ: cùng một bước hiện cùng lúc ở hai hàng.
Khung tĩnh cho prefers-reduced-motion là trạng thái cuối.

Chạy: python3 make_continuous_batching.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

SLOTS = 3
FIRST = [("A", 3), ("B", 6), ("C", 2)]   # có sẵn lúc đầu
QUEUE = [("D", 3), ("E", 2)]             # đang chờ


def simulate(continuous: bool) -> list[list[str | None]]:
    """Trả về grid[slot][step] = tên request hoặc None (slot rảnh)."""
    pending = list(QUEUE)
    slots: list[list] = [[name, n] for name, n in FIRST]
    grid: list[list[str | None]] = [[] for _ in range(SLOTS)]
    while any(s for s in slots) or pending:
        # static: chỉ nhận request mới khi cả batch đã xong
        can_admit = continuous or not any(s for s in slots)
        for i in range(SLOTS):
            if can_admit and slots[i] is None and pending:
                name, n = pending.pop(0)
                slots[i] = [name, n]
        for i in range(SLOTS):
            if slots[i] is None:
                grid[i].append(None)
            else:
                grid[i].append(slots[i][0])
                slots[i][1] -= 1
                if slots[i][1] == 0:
                    slots[i] = None
    return grid


def finish(grid: list[list[str | None]]) -> int:
    return max(max(j + 1 for j, v in enumerate(row) if v) for row in grid)


def idle(grid: list[list[str | None]], steps: int) -> int:
    return sum(1 for row in grid for v in row[:steps] if v is None)


G_STATIC, G_CONT = simulate(False), simulate(True)
N_STATIC, N_CONT = finish(G_STATIC), finish(G_CONT)
assert (N_STATIC, N_CONT) == (9, 6), (N_STATIC, N_CONT)
assert [r[:3] for r in G_CONT] == [["A"] * 3, ["B"] * 3, ["C", "C", "D"]]
assert G_CONT[0][3] == "E"                 # E vào ngay khi A xong
assert G_STATIC[0][3:6] == [None] * 3      # slot của A ngồi không tới bước 6
NSTEP = N_STATIC
for g in (G_STATIC, G_CONT):
    for row in g:
        row += [None] * (NSTEP - len(row))

# Nhịp: mỗi bước 1,4 giây; kết luận giữ ít nhất 3 giây
DUR = 20.0
a = svgkit.Anim(dur=DUR, fade_out=95)
T0, STEP = 1.0, 1.4
T_END = T0 + NSTEP * STEP
HOLD = DUR * a.fade_out / 100 - T_END
assert STEP >= 1.0 and HOLD >= 3.0, HOLD


def pct(t: float) -> float:
    return round(t / DUR * 100, 2)


def show(t: float, inner: str) -> str:
    return f'<g {a([(pct(t), 0), (pct(t + 0.4), 1)])}>{inner}</g>'


CW, CH, GAP = 60, 30, 6
X0 = 200
W = X0 + NSTEP * (CW + GAP) - GAP + 40
RS, RC = 110, 276                       # y đầu của hai hàng
H = RC + SLOTS * (CH + GAP) + 90
NEW = {name for name, _ in QUEUE}


def cell_x(j: int) -> float:
    return X0 + j * (CW + GAP)


body: list[str] = []
body.append('<text class="title" x="20" y="38">Static batching và continuous batching</text>')
body.append(svgkit.legend(20, 70, [("blue", "A, B, C có sẵn"), ("orange", "D, E đang chờ"),
                                    ("slot", "slot rảnh")]))

for j in range(NSTEP):
    body.append(f'<text class="sub mid" x="{cell_x(j) + CW / 2}" y="{RS - 10}">bước {j + 1}</text>')

rows = ((RS, "Static", "đợi cả batch xong", G_STATIC, N_STATIC),
        (RC, "Continuous", "thay request ngay", G_CONT, N_CONT))
for y0, name, sub, grid, n_end in rows:
    body.append(f'<text class="row" x="20" y="{y0 + 22}">{name}</text>')
    body.append(f'<text class="sub" x="20" y="{y0 + 40}">{sub}</text>')
    for i in range(SLOTS):
        y = y0 + i * (CH + GAP)
        body.append(f'<text class="sub end" x="{X0 - 10}" y="{y + CH / 2 + 4}">slot {i + 1}</text>')
        for j in range(NSTEP):
            body.append(f'<rect class="slot" x="{cell_x(j)}" y="{y}" width="{CW}" height="{CH}" rx="5"/>')

# Mỗi bước hiện cùng lúc ở hai hàng
for j in range(NSTEP):
    parts = []
    for y0, _, _, grid, _ in rows:
        for i in range(SLOTS):
            v = grid[i][j]
            if v is None:
                continue
            y = y0 + i * (CH + GAP)
            cls = "orange" if v in NEW else "blue"
            parts.append(f'<rect class="{cls}" x="{cell_x(j)}" y="{y}" width="{CW}" height="{CH}" rx="5"/>'
                         f'<text class="lbl" x="{cell_x(j) + CW / 2}" y="{y + CH / 2 + 5}">{v}</text>')
    body.append(show(T0 + j * STEP, "".join(parts)))

# Mốc "xong" cho mỗi hàng: vạch sau bước cuối, nhãn bên dưới hàng
for y0, _, _, grid, n_end in rows:
    xe = cell_x(n_end - 1) + CW + GAP / 2
    yb = y0 + SLOTS * (CH + GAP) - GAP
    body.append(show(T0 + (n_end - 1) * STEP + 0.6,
                     f'<line class="lfg" x1="{xe}" y1="{y0 - 4}" x2="{xe}" y2="{yb + 6}" style="stroke-width:2"/>'
                     f'<text class="cap end" x="{xe}" y="{yb + 24}" style="font-weight:600">'
                     f'xong sau {n_end} bước</text>'))

# Kết luận, hiện sau bước cuối
CONC = "Request xong rời batch ngay, request chờ vào thế chỗ ở bước kế tiếp."
assert len(CONC) / HOLD <= 15, len(CONC) / HOLD
body.append(show(T_END, f'<text class="cap" x="20" y="{H - 20}">{CONC}</text>'))

svgkit.write(__file__, "continuous-batching.svg", svgkit.svg(
    W, H, "Ba slot, request A, B, C cần 3, 6, 2 bước, D và E đang chờ. Static batching đợi B xong ở "
    "bước 6 mới nhận D, E, xong sau 9 bước, slot của A và C ngồi không. Continuous batching cho D, E "
    "vào ngay khi C, A xong, xong sau 6 bước", body, a))
