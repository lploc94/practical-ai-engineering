"""Sinh breaker-timeline.svg: SVG tĩnh so chuỗi fallback có và không có circuit breaker (§3.3).

Tham số lấy từ ví dụ ở §3.1 (script kiểm README còn ghi đúng các số này):
Breaker(threshold=5, cooldown_s=30), attempts_per_target=2.
Logic breaker mô phỏng lại đúng class `Breaker` của §3.1: mở khi số lỗi liên tiếp
đạt ngưỡng, chặn tới hết cooldown, hết cooldown thì cho một request thăm dò.

Kịch bản giả định, chỉ để minh hoạ thứ tự:
- Model chính trả 529 từ giây 0 tới giây 35, sau đó hồi phục.
- Cứ 3 giây có một request (giây 0, 3, ..., 42). Model fallback luôn trả lời được.
- Hai lần thử ở model chính cách nhau 0,5 s (backoff); thời gian phản hồi bỏ qua.
Kết quả:
- Không có breaker: mọi request trước giây 35 thử model chính 2 lần, lỗi cả hai,
  rồi mới sang fallback: 24 lần thử thất bại, người dùng nào cũng chịu backoff.
- Có breaker: lỗi thứ 5 (request ở giây 6) mở breaker tới giây 36; các request
  giữa chừng đi thẳng tới fallback; giây 36 một request thăm dò thành công nên
  breaker đóng lại. Tổng 5 lần thử thất bại.
Hình tĩnh: cả hai dòng thời gian cần nhìn cùng lúc để so, không có ý nào mất khi
bỏ chuyển động.

Chạy: python3 make_breaker_timeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
THRESHOLD, COOLDOWN, PER_TARGET = 5, 30, 2
assert f"Breaker(threshold={THRESHOLD}, cooldown_s={COOLDOWN})" in README
assert f"attempts_per_target={PER_TARGET}" in README
RECOVER = 35                     # giả định: model chính hồi phục ở giây 35
STEP = 3                         # giả định: mỗi 3 giây một request
REQS = list(range(0, 43, STEP))
BACKOFF = 0.5


def simulate(use_breaker: bool):
    """Trả về danh sách (t, [kết quả các lần thử ở model chính], dùng fallback?)."""
    failures, open_until, probing = 0, 0.0, False
    out = []
    for t in REQS:
        tries: list[str] = []
        served_primary = False
        for i in range(PER_TARGET):
            now = t + i * BACKOFF
            if use_breaker and failures >= THRESHOLD:
                if now < open_until or probing:
                    tries.append("skip")
                    break
                probing = True
            if now < RECOVER:
                tries.append("fail")
                failures += 1
                probing = False
                if failures >= THRESHOLD:
                    open_until = now + COOLDOWN
                continue
            tries.append("ok")
            failures, probing = 0, False
            served_primary = True
            break
        out.append((t, tries, not served_primary))
    return out


NO_BR, BR = simulate(False), simulate(True)
fails = lambda rows: sum(tr.count("fail") for _, tr, _ in rows)  # noqa: E731
assert fails(NO_BR) == 24 and fails(BR) == THRESHOLD
OPEN_AT = 6
assert [t for t, tr, _ in BR if "skip" in tr][0] == OPEN_AT          # lần thử 2 của request giây 6 bị chặn
PROBE = OPEN_AT + COOLDOWN
assert PROBE == 36 and [tr for t, tr, _ in BR if t == PROBE][0] == ["ok"]
assert all(tr == ["skip"] for t, tr, _ in BR if OPEN_AT < t < PROBE)
# Đoạn văn §3.3 cạnh hình phải ghi đúng kịch bản và kết quả này
assert f"model chính lỗi {RECOVER} giây, mỗi {STEP} giây một request" in README
assert f"{sum(1 for t, _, _ in NO_BR if t < RECOVER)} request đầu đều thử model chính hai lần, tổng {fails(NO_BR)} lần thử thất bại" in README
assert f"chỉ có {fails(BR)} lần thử thất bại" in README

W, H = 980, 545
X0, X1 = 200, 930
TEND = 44
CW = 18                                     # độ rộng ô một lần thử


def x(s: float) -> float:
    return X0 + s / TEND * (X1 - X0)


body: list[str] = []
body.append('<text class="title" x="20" y="38">Circuit breaker: request đi thẳng tới fallback khi model chính đang sập</text>')
body.append(svgkit.legend(20, 66, [("hatch", "thử model chính, lỗi 529"), ("blue", "được phục vụ"),
                                    ("slot", "bỏ qua model chính")]))


def panel(y0: float, title: str, sub: str, rows) -> None:
    yp, yf = y0 + 36, y0 + 108
    body.append(f'<text class="row" x="20" y="{y0 + 10}">{title}</text>')
    body.append(f'<text class="sub" x="20" y="{y0 + 27}">{sub}</text>')
    body.append(f'<text class="cap" x="20" y="{yp + 24}">Model chính</text>')
    body.append(f'<text class="cap" x="20" y="{yf + 15}">Model fallback</text>')
    for t, tries, fb in rows:
        cx = x(t)
        for i, r in enumerate(tries):
            yy = yp + i * 20
            cls = {"fail": "hatch", "ok": "blue", "skip": "slot"}[r]
            body.append(f'<rect class="{cls}" x="{cx - CW / 2:.1f}" y="{yy}" width="{CW}" height="16" rx="3"/>')
        if fb:
            body.append(f'<rect class="blue" x="{cx - CW / 2:.1f}" y="{yf}" width="{CW}" height="20" rx="3"/>')


panel(92, "Không có breaker", f"mỗi request thử model chính {PER_TARGET} lần rồi mới sang fallback", NO_BR)
panel(262, "Có breaker", f"threshold={THRESHOLD}, cooldown_s={COOLDOWN}", BR)

# Trạng thái breaker dưới panel 2
yb = 262 + 162
ay = yb + 46
# Đường hồi phục vẽ trước nhãn để halo của nhãn che được nét đứt
body.append(f'<line class="grid dash" x1="{x(RECOVER):.1f}" y1="110" x2="{x(RECOVER):.1f}" y2="{ay}"/>')
body.append(f'<text class="cap" x="20" y="{yb + 15}">Breaker</text>')
seg = [(0, OPEN_AT, "đóng"), (OPEN_AT, PROBE, f"mở {COOLDOWN} s: không gửi tới model chính"),
       (PROBE, TEND, "đóng")]
for a, b, lab in seg:
    cls = "lfg" if "mở" in lab else "lmuted"
    body.append(f'<path class="{cls}" d="M{x(a) + 3:.1f},{yb}V{yb + 8}H{x(b) - 3:.1f}V{yb}"/>'
                f'<text class="sub mid" x="{(x(a) + x(b)) / 2:.1f}" y="{yb + 24}">{lab}</text>')
body.append(f'<text class="sub mid halo" x="{x(PROBE):.1f}" y="{yb - 6}">thăm dò</text>')
body.append(f'<text class="sub mid halo" x="{x(OPEN_AT):.1f}" y="{yb - 6}">lỗi thứ {THRESHOLD}</text>')

# Trục thời gian
body.append(f'<line class="axis" x1="{X0}" y1="{ay}" x2="{X1}" y2="{ay}"/>')
for s in range(0, TEND + 1, 6):
    body.append(f'<line class="axis" x1="{x(s):.1f}" y1="{ay}" x2="{x(s):.1f}" y2="{ay + 5}"/>'
                f'<text class="sub mid" x="{x(s):.1f}" y="{ay + 18}">{s} s</text>')
body.append(f'<text class="sub halo" x="{x(RECOVER) + 6:.1f}" y="104">model chính hồi phục</text>')

body.append(f'<text class="cap2" x="20" y="{H - 16}">Không breaker: {fails(NO_BR)} lần thử thất bại, request nào cũng chờ '
            f'backoff. Có breaker: {fails(BR)}. Kịch bản là giả định.</text>')

svgkit.write(__file__, "breaker-timeline.svg", svgkit.svg(
    W, H, f"Model chính lỗi 529 trong 35 giây, mỗi 3 giây có một request. Không có breaker, request nào cũng thử "
    f"model chính {PER_TARGET} lần rồi mới sang fallback, tổng {fails(NO_BR)} lần thử thất bại. Có breaker với ngưỡng "
    f"{THRESHOLD} và cooldown {COOLDOWN} giây, lỗi thứ {THRESHOLD} mở breaker, request sau đó đi thẳng tới fallback, "
    f"hết cooldown một request thăm dò thành công và breaker đóng lại", body))
