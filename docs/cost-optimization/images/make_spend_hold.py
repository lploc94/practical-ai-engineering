"""Sinh spend-hold.svg: vì sao hạn mức đô la phải đếm cả phần giữ chỗ (§3.5).

Tình huống đúng như test 3 trong scripts/t_budget.py (mục 4 của danh sách test ở
§3.5): hạn mức 1,00 USD một ngày, tenant đã tiêu 0,10 USD, một request đang chạy đã
giữ chỗ 0,80 USD (chưa có chi phí thật), request mới cần giữ chỗ 0,20 USD.
- Chỉ đếm số đã tiêu: 0,10 + 0,20 ≤ 1,00 nên cho qua; nếu cả hai request dùng hết
  phần tối đa thì tổng là 1,10 USD, vượt hạn mức.
- SpendLedger đếm đã tiêu cộng đang giữ chỗ: 0,10 + 0,80 + 0,20 = 1,10 > 1,00 nên
  ném BudgetExceeded.
Điều kiện chặn trong SpendLedger.hold là used + max_cost > limit (dấu lớn hơn ngặt).

Chạy: python3 make_spend_hold.py
"""

import sys
from decimal import Decimal as D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

LIMIT, SPENT, HELD, NEW = D("1.00"), D("0.10"), D("0.80"), D("0.20")
naive_pass = SPENT + NEW <= LIMIT
ledger_pass = not (SPENT + HELD + NEW > LIMIT)
assert naive_pass and not ledger_pass
assert SPENT + HELD + NEW == D("1.10")
assert not (HELD + NEW > LIMIT)   # nếu chưa tiêu gì thì 0,80 + 0,20 = 1,00 vẫn được cho qua


def vn(v: D) -> str:
    return f"{v:.2f}".replace(".", ",")


W, H = 820, 380
X0, X1 = 220, 720                  # trục 0..1,2 USD
XMAX = D("1.2")
BH = 40
Y1, Y2 = 110, 222


def x(v: D) -> float:
    return X0 + float(v / XMAX) * (X1 - X0)


def seg(y: float, a: D, b: D, cls: str, label: str) -> str:
    x0, x1 = x(a) + 1, x(b) - 1
    tx = (x0 + x1) / 2
    if a < LIMIT < b:              # nhãn nằm bên trái vạch hạn mức để không bị vạch cắt
        tx = (x0 + x(LIMIT)) / 2
    return (f'<rect class="{cls}" x="{x0:.1f}" y="{y}" width="{x1 - x0:.1f}" height="{BH}" rx="5"/>'
            f'<text class="lbl" style="font-size:13px" x="{tx:.1f}" y="{y + 25}">{label}</text>')


body: list[str] = []
body.append('<text class="title" x="20" y="38">Hạn mức phải đếm cả phần đang giữ chỗ</text>')
body.append(svgkit.legend(20, 70, [("blue", "đã tiêu"), ("orange", "request đang chạy"),
                                    ("slot", "request mới, chưa chạy")]))

# Hàng 1: chỉ đếm số đã tiêu
body.append(f'<text class="row" x="20" y="{Y1 + 17}">Chỉ đếm đã tiêu</text>')
body.append(f'<text class="sub" x="20" y="{Y1 + 34}">0,10 + 0,20 ≤ 1,00: cho qua</text>')
body.append(seg(Y1, D(0), SPENT, "blue", vn(SPENT)))
body.append(seg(Y1, SPENT, SPENT + HELD, "orange", f"{vn(HELD)} không được đếm"))
body.append(seg(Y1, SPENT + HELD, SPENT + HELD + NEW, "orange", vn(NEW)))
body.append(f'<text class="cap" x="{x(SPENT + HELD + NEW) + 8:.1f}" y="{Y1 + 25}">có thể tới {vn(SPENT + HELD + NEW)}</text>')

# Hàng 2: SpendLedger
body.append(f'<text class="row" x="20" y="{Y2 + 17}">SpendLedger</text>')
body.append(f'<text class="sub" x="20" y="{Y2 + 34}">0,10 + 0,80 + 0,20 &gt; 1,00: chặn</text>')
body.append(seg(Y2, D(0), SPENT, "blue", vn(SPENT)))
body.append(seg(Y2, SPENT, SPENT + HELD, "orange", f"giữ chỗ {vn(HELD)}"))
body.append(seg(Y2, SPENT + HELD, SPENT + HELD + NEW, "slot", vn(NEW)))
body.append(f'<text class="cap" x="{x(SPENT + HELD + NEW) + 8:.1f}" y="{Y2 + 18}" style="font-weight:600">'
            'bị chặn</text>')
body.append(f'<text class="sub" x="{x(SPENT + HELD + NEW) + 8:.1f}" y="{Y2 + 34}" '
            f'style="font-family:{svgkit.MONO}">BudgetExceeded</text>')

# Vạch hạn mức
lx = x(LIMIT)
body.append(f'<line class="lfg" x1="{lx:.1f}" y1="{Y1 - 16}" x2="{lx:.1f}" y2="{Y2 + BH + 12}" style="stroke-width:2.5"/>')
body.append(f'<text class="sub mid" x="{lx:.1f}" y="{Y1 - 22}">hạn mức {vn(LIMIT)} USD</text>')

# Trục
ay = Y2 + BH + 32
body.append(f'<line class="axis" x1="{X0}" y1="{ay}" x2="{X1}" y2="{ay}"/>')
for v in (D(0), D("0.25"), D("0.5"), D("0.75"), D(1)):
    body.append(f'<line class="axis" x1="{x(v):.1f}" y1="{ay}" x2="{x(v):.1f}" y2="{ay + 5}"/>'
                f'<text class="sub mid" x="{x(v):.1f}" y="{ay + 18}">{vn(v)}</text>')
body.append(f'<text class="sub" x="{X1 - 30}" y="{ay + 18}">USD</text>')

body.append(f'<text class="cap2" x="20" y="{H - 30}">Request đang chạy chưa có chi phí thật nhưng có thể tiêu tới phần đã '
            'giữ chỗ. Bỏ qua nó thì hai request</text>')
body.append(f'<text class="cap2" x="20" y="{H - 12}">đồng thời cùng qua kiểm tra và tổng chi tiêu có thể tới '
            f'{vn(SPENT + HELD + NEW)} USD trên hạn mức {vn(LIMIT)}.</text>')

svgkit.write(__file__, "spend-hold.svg", svgkit.svg(
    W, H, f"Hạn mức {vn(LIMIT)} USD, đã tiêu {vn(SPENT)}, một request đang chạy giữ chỗ {vn(HELD)}. Chỉ đếm số đã "
    f"tiêu thì request mới {vn(NEW)} USD được cho qua và tổng có thể tới {vn(SPENT + HELD + NEW)} USD; SpendLedger "
    "đếm cả phần giữ chỗ nên chặn request mới bằng BudgetExceeded", body))
