"""Sinh coordinated-omission.svg: SVG tĩnh, closed-loop và open-loop đo cùng một lần server khựng (§1.3).

Mô phỏng tất định, số liệu giả định chỉ để minh hoạ:
- Lịch gửi: 1 request mỗi giây, ở giây 0..9 (10 request).
- Server xử lý tuần tự từng request, mỗi request 0,5 giây; từ giây 2 tới giây 6,5
  server khựng 4,5 giây (ví dụ GC, swap), không xử lý gì.
- Closed-loop: client muốn gửi theo cùng lịch nhưng chỉ gửi khi đã nhận
  response của request trước, đúng kiểu tester mà README của wrk2 [1] mô tả. Các
  mốc lịch trôi qua trong lúc chờ thì không gửi (đây là phần bị "omit"), độ trễ
  tính từ lúc request thật sự rời client. (wrk không có nhịp, gửi ngay sau
  response; trong lần khựng nó cũng chỉ đo được 1 request chậm.)
- Open-loop: gửi đúng lịch dù request trước chưa xong; độ trễ tính từ thời điểm
  lẽ ra được gửi theo lịch (§1.3, điều 2). Request xếp hàng sau lần khựng.
"Chậm" nghĩa là độ trễ trên 1 giây. Script tự mô phỏng rồi assert các số in trên hình.
Hình tĩnh vì cần so hai cách đo cạnh nhau trên cùng trục thời gian; thứ tự sự
kiện đã đọc được từ trục, chuyển động không thêm ý.

Chạy: python3 make_coordinated_omission.py
"""

import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
assert "coordinated omission" in README and "lẽ ra* được gửi theo lịch" in README

TICKS = list(range(10))          # lịch gửi: giây 0..9
SERVICE = 0.5
STALL = (2.0, 6.5)
SLOW = 1.0


def serve(arrival: float, free_at: float) -> float:
    """Server tuần tự; thời gian khựng không xử lý được gì. Trả thời điểm xong."""
    start = max(arrival, free_at)
    if STALL[0] <= start < STALL[1]:
        start = STALL[1]
    end = start + SERVICE
    if start < STALL[0] < end:   # đang xử lý thì khựng: phần còn lại chờ tới hết khựng
        end = STALL[1] + (end - STALL[0])
    return end


# Closed-loop: gửi ở mốc lịch đầu tiên không sớm hơn lúc nhận response trước
closed: dict[int, tuple[float, float]] = {}      # mốc -> (gửi, xong)
free = 0.0
done_prev = 0.0
for k in TICKS:
    if k < done_prev:
        continue                                   # mốc đã qua trong lúc chờ: không gửi
    end = serve(k, free)
    free = end
    closed[k] = (k, end)
    done_prev = end
omitted = [k for k in TICKS if k not in closed]

# Open-loop: gửi đúng lịch, đo từ mốc lịch
opened: dict[int, tuple[float, float]] = {}
free = 0.0
for k in TICKS:
    end = serve(k, free)
    free = end
    opened[k] = (k, end)

lat_c = [e - s for s, e in closed.values()]
lat_o = [e - s for s, e in opened.values()]
slow_c = sum(x > SLOW for x in lat_c)
slow_o = sum(x > SLOW for x in lat_o)
med_c, med_o = statistics.median(lat_c), statistics.median(lat_o)
assert omitted == [3, 4, 5, 6], omitted
assert (len(lat_c), slow_c, med_c) == (6, 1, 0.5), (len(lat_c), slow_c, med_c)
assert (len(lat_o), slow_o, med_o) == (10, 8, 2.75), (len(lat_o), slow_o, med_o)
assert max(lat_c) == max(lat_o) == 5.0
assert opened[9][1] == 10.5


def fmt(s: float) -> str:
    return f"{s:g}".replace(".", ",")


W, H = 940, 610
X0, X1 = 190, 900                # trục 0 → TMAX giây
TMAX = 11
RH, RG = 14, 4                   # chiều cao một request, khoảng cách
YC, YO = 130, 330                # đỉnh hai panel


def x(s: float) -> float:
    return X0 + s / TMAX * (X1 - X0)


body: list[str] = []
body.append('<text class="title" x="20" y="38">Coordinated omission: closed-loop bỏ qua đúng lúc server chậm</text>')
body.append(svgkit.legend(20, 70, [("blue", f"request đo được, ≤ {fmt(SLOW)} s"),
                                    ("hatch", f"request đo được, chậm > {fmt(SLOW)} s"),
                                    ("slot", "mốc lịch không gửi")]))

# Dải server khựng, phủ cả hai panel
xs0, xs1 = x(STALL[0]), x(STALL[1])
body.append(f'<rect x="{xs0:.1f}" y="{YC - 30}" width="{xs1 - xs0:.1f}" height="{YO + 10 * (RH + RG) + 4 - (YC - 30)}" '
            f'style="fill:var(--tok)"/>')
body.append(f'<text class="sub mid" x="{(xs0 + xs1) / 2:.1f}" y="{YC - 14}">server khựng {fmt(STALL[1] - STALL[0])} s</text>')


def panel(y0: float, title: str, sub: list[str], reqs: dict[int, tuple[float, float]], summary: list[str]) -> None:
    body.append(f'<text class="row" x="20" y="{y0 + 14}">{title}</text>')
    for i, line in enumerate(sub):
        body.append(f'<text class="sub" x="20" y="{y0 + 32 + 16 * i}">{line}</text>')
    for k in TICKS:
        yy = y0 + k * (RH + RG)
        if k in reqs:
            s, e = reqs[k]
            cls = "hatch" if e - s > SLOW else "blue"
            body.append(f'<rect class="{cls}" x="{x(s):.1f}" y="{yy}" width="{x(e) - x(s):.1f}" height="{RH}" rx="3" '
                        'style="stroke-width:1.5"/>')
        else:
            body.append(f'<rect class="slot" x="{x(k):.1f}" y="{yy}" width="{x(k + SERVICE) - x(k):.1f}" '
                        f'height="{RH}" rx="3"/>')
    for i, line in enumerate(summary):
        body.append(f'<text class="cap" x="20" y="{y0 + 84 + 18 * i}" style="font-weight:600">{line}</text>')


panel(YC, "Closed-loop", ["theo lịch, nhưng chờ", "response mới gửi tiếp"], closed,
      [f"{len(lat_c)} đo được, {slow_c} chậm", f"trung vị {fmt(med_c)} s"])
body.append(f'<text class="sub halo" x="{x(omitted[0] + SERVICE) + 8:.1f}" '
            f'y="{YC + omitted[0] * (RH + RG) + 11}">{len(omitted)} mốc lịch không gửi</text>')
panel(YO, "Open-loop", ["gửi đúng lịch, đo từ", "lúc lẽ ra được gửi"], opened,
      [f"{len(lat_o)} đo được, {slow_o} chậm", f"trung vị {fmt(med_o)} s"])
body.append(f'<text class="sub halo end" x="{x(opened[9][0]) - 6:.1f}" y="{YO + 9 * (RH + RG) + 11}">'
            f'request cuối xong ở giây {fmt(opened[9][1])}</text>')

# Trục thời gian
ay = YO + 10 * (RH + RG) + 22
body.append(f'<line class="axis" x1="{X0}" y1="{ay}" x2="{X1}" y2="{ay}"/>')
for s in range(0, TMAX + 1):
    body.append(f'<line class="axis" x1="{x(s):.1f}" y1="{ay}" x2="{x(s):.1f}" y2="{ay + 5}"/>'
                f'<text class="sub mid" x="{x(s):.1f}" y="{ay + 18}">{s} s</text>')

body.append(f'<text class="cap2" x="20" y="{H - 30}">Giả định: lịch 1 request/giây, server xử lý tuần tự 0,5 s mỗi request, '
            f'khựng từ giây {fmt(STALL[0])} tới {fmt(STALL[1])}.</text>')
body.append(f'<text class="cap2" x="20" y="{H - 12}">Mỗi hàng là một request: thanh chạy từ lúc gửi (open-loop: lúc lẽ ra '
            'được gửi) tới lúc nhận response.</text>')

assert X1 <= W - 30                                          # lề phải ≥ 30 px

ALT = (f"Cùng một lần server khựng {fmt(STALL[1] - STALL[0])} giây: closed-loop chỉ gửi tiếp khi có response nên bỏ "
    f"{len(omitted)} mốc lịch và chỉ thấy {slow_c} request chậm, trung vị {fmt(med_c)} giây; open-loop gửi đúng lịch, "
    f"đo từ lúc lẽ ra được gửi, thấy {slow_o} trên {len(lat_o)} request chậm, trung vị {fmt(med_o)} giây")
assert f"![{ALT}](images/coordinated-omission.svg)" in README   # alt trong bài khớp số mô phỏng
svgkit.write(__file__, "coordinated-omission.svg", svgkit.svg(W, H, ALT, body))
