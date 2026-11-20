"""Sinh group-split.svg: SVG tĩnh, chia ngẫu nhiên từng dòng so với chia theo nhóm (§3.1, §3.3).

Dữ liệu thật: scripts/toy.jsonl (576 dòng, 48 mẫu câu) qua scripts/ft_data.py.
- Bên phải: prepare(group_of=mẫu câu, test=0.2, val=0.1) như train_lora.py, ra đúng
  361 train, 101 validation, 60 test ở §3.3.
- Bên trái: cùng các dòng đã loại trùng, nhưng khoá chia là dấu vân tay của từng dòng
  (split_of(prompt_key(...))), tức chia ngẫu nhiên từng dòng theo hash, cùng tỉ lệ.
Mỗi ô là một mẫu câu, mỗi chấm là một dòng, tô theo tập nó rơi vào. Hai màu có nghĩa:
train xanh dương, test cam; validation màu trung tính.
Hình tĩnh vì đây là kết quả chia, không có diễn biến theo thời gian.

Chạy: python3 make_group_split.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "assets"))
sys.path.insert(0, str(HERE.parent / "scripts"))
import svgkit  # noqa: E402
from ft_data import load_jsonl, prepare, prompt_key, split_of  # noqa: E402

records, errors = load_jsonl(HERE.parent / "scripts" / "toy.jsonl")
assert len(records) == 576 and not errors
splits, report = prepare(records, group_of=lambda r: r["template"], test=0.2, val=0.1)
assert {k: len(v) for k, v in splits.items()} == {"train": 361, "val": 101, "test": 60}
assert len(report["duplicate"]) == 54

by_group = {r["template"]: s for s, rows in splits.items() for r in rows}
kept = [r for rows in splits.values() for r in rows]
by_row = [(r["template"], split_of(prompt_key(r), 0.2, 0.1)) for r in kept]

INTENTS = ["track", "cancel", "return", "warranty", "payment", "other"]
group_cells: dict[str, list[str]] = defaultdict(list)
row_cells: dict[str, list[str]] = defaultdict(list)
for r in kept:
    group_cells[r["template"]].append(by_group[r["template"]])
for t, s in by_row:
    row_cells[t].append(s)
assert len(group_cells) == 48

# Chia theo nhóm: mỗi mẫu câu nằm trọn trong một tập; test = 5 mẫu câu, không có track/other
test_groups = sorted(t for t, s in by_group.items() if s == "test")
assert len(test_groups) == 5 and not any(t.split("/")[0] in ("track", "other") for t in test_groups)
assert all(len(set(v)) == 1 for v in group_cells.values())
# Chia từng dòng: đếm mẫu câu có dòng ở cả train và test
leak = sum(1 for v in row_cells.values() if "train" in v and "test" in v)
row_counts = Counter(s for _, s in by_row)
test_rows_leaky = sum(1 for t, s in by_row if s == "test" and "train" in row_cells[t])
assert leak == 43 and test_rows_leaky == row_counts["test"] == 110, (leak, test_rows_leaky, row_counts)

ORDER = {"train": 0, "val": 1, "test": 2}
FILL = {"train": "fblue", "test": "forange", "val": None}

W, H = 930, 512
CW, CH_, GAP = 40, 46, 6          # ô mẫu câu
DOT, DG = 7, 2                     # chấm = một dòng, lưới 4 cột x 3 hàng
LX = 100                           # lề trái cho tên intent
PANEL = 8 * (CW + GAP)
P1, P2 = LX, LX + PANEL + 60
TOP = 128

body: list[str] = []
body.append('<text class="title" x="20" y="38">Chia train/test theo từng dòng hay theo mẫu câu</text>')
body.append('<text class="cap2" x="20" y="62">Dữ liệu đồ chơi §3.3: 522 dòng sau khi loại trùng, 48 mẫu câu. '
            'Mỗi ô là một mẫu câu, mỗi chấm là một dòng.</text>')
# Chú giải vẽ tay: ô validation tô đặc var(--slot) giống chấm dữ liệu
lx = 20
for attr, text in (('class="fblue"', "train"), ('style="fill:var(--slot)"', "validation"), ('class="forange"', "test")):
    body.append(f'<rect {attr} x="{lx}" y="79" width="14" height="14" rx="3"/><text class="sub" x="{lx + 20}" y="90">{text}</text>')
    lx += 20 + 7 * len(text) + 24


def panel(x0: float, cells: dict[str, list[str]], title: str) -> None:
    body.append(f'<text class="row" x="{x0}" y="{TOP - 12}">{title}</text>')
    for i, intent in enumerate(INTENTS):
        y = TOP + i * (CH_ + GAP)
        if x0 == P1:
            body.append(f'<text class="sub end" x="{x0 - 10}" y="{y + CH_ / 2 + 4}">{intent}</text>')
        for j in range(8):
            x = x0 + j * (CW + GAP)
            body.append(f'<rect class="slot" x="{x}" y="{y}" width="{CW}" height="{CH_}" rx="5"/>')
            for k, s in enumerate(sorted(cells[f"{intent}/{j}"], key=ORDER.get)):
                dx = x + 4 + (k % 4) * (DOT + DG)
                dy = y + 4 + (k // 4) * (DOT + DG)
                cls = FILL[s]
                attr = f'class="{cls}"' if cls else 'style="fill:var(--slot)"'
                body.append(f'<rect {attr} x="{dx}" y="{dy}" width="{DOT}" height="{DOT}" rx="1.5"/>')


panel(P1, row_cells, "Theo từng dòng")
panel(P2, group_cells, "Theo mẫu câu (như §3.3)")

yb = TOP + 6 * (CH_ + GAP) + 22
body.append(f'<text class="cap" x="{P1}" y="{yb}">{leak}/48 mẫu câu có dòng ở cả train lẫn test.</text>')
body.append(f'<text class="cap2" x="{P1}" y="{yb + 20}">Cả {test_rows_leaky} dòng test đều có dòng cùng mẫu câu trong train:</text>')
body.append(f'<text class="cap2" x="{P1}" y="{yb + 38}">điểm test đo khả năng nhớ.</text>')
body.append(f'<text class="cap" x="{P2}" y="{yb}">Test: 5 mẫu câu, 60 dòng, chưa thấy khi train.</text>')
body.append(f'<text class="cap2" x="{P2}" y="{yb + 20}">Nhưng chỉ 5 cụm, và không có track hay other:</text>')
body.append(f'<text class="cap2" x="{P2}" y="{yb + 38}">kiểm số lượng mỗi lớp sau khi chia.</text>')

svgkit.write(__file__, "group-split.svg", svgkit.svg(
    W, H, f"Dữ liệu đồ chơi 522 dòng, 48 mẫu câu. Chia ngẫu nhiên từng dòng: {leak}/48 mẫu câu có dòng ở cả train "
    "lẫn test, nên điểm test đo khả năng nhớ. Chia theo mẫu câu: mỗi mẫu câu nằm trọn một tập, test gồm 5 mẫu câu "
    "(60 dòng) chưa thấy khi train, nhưng không có intent track hay other", body))
