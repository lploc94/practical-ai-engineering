"""Sinh idle-heartbeat.svg: SVG tĩnh về khoảng im lặng và idle timeout (§3.2).

Số liệu giả định: reasoning model im lặng 75 giây trước token đầu tiên; tầng giữa
có idle timeout 60 giây (mặc định của nginx proxy_read_timeout và AWS ALB
idle_timeout.timeout_seconds, xem §3.2). Backend gửi comment ": ping" mỗi 15 giây.
- Không heartbeat: sau 60 giây không có byte nào, kết nối bị đóng; token bắt đầu từ
  giây 75 không bao giờ tới client.
- Có heartbeat: khoảng im lặng dài nhất là 15 giây, dưới ngưỡng 60 giây.
Hình tĩnh vì ý chính là so độ dài khoảng im lặng với ngưỡng, không phải diễn biến.

Chạy: python3 make_idle_heartbeat.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

SILENCE, IDLE, PING, TEND = 75, 60, 15, 100
pings = list(range(PING, SILENCE, PING))           # 15, 30, 45, 60
assert pings == [15, 30, 45, 60]
events = [0] + pings + [SILENCE]                    # header ở giây 0, ping, token đầu
gaps = [b - a for a, b in zip(events, events[1:])]
assert max(gaps) == PING < IDLE < SILENCE

W, H = 820, 352
X0, X1 = 200, 790
BH = 30
YA, YB = 104, 214


def x(s: float) -> float:
    return X0 + s / TEND * (X1 - X0)


body: list[str] = []
body.append('<text class="title" x="20" y="38">Im lặng lâu hơn idle timeout thì kết nối bị đóng</text>')
body.append(svgkit.legend(20, 66, [("blue", "token gửi xuống client"), ("hatch", "bị cắt, không tới client")]))

# Hàng A: không heartbeat
body.append(f'<text class="row" x="20" y="{YA + 14}">Không heartbeat</text>')
body.append(f'<text class="sub" x="20" y="{YA + 31}">im lặng {SILENCE} s</text>')
body.append(f'<rect class="slot" x="{X0}" y="{YA}" width="{x(IDLE) - X0:.1f}" height="{BH}" rx="5"/>')
body.append(f'<text class="cap2 mid" x="{(X0 + x(IDLE)) / 2:.1f}" y="{YA + 20}">model đang suy luận, không có byte nào</text>')
body.append(f'<rect class="hatch" x="{x(SILENCE):.1f}" y="{YA}" width="{X1 - x(SILENCE):.1f}" height="{BH}" rx="5"/>')
body.append(f'<line class="lred" x1="{x(IDLE):.1f}" y1="{YA - 12}" x2="{x(IDLE):.1f}" y2="{YA + BH + 12}" '
            'style="stroke-width:3"/>')
body.append(f'<text class="sub halo" x="{x(IDLE) + 6:.1f}" y="{YA - 4}">giây {IDLE}: đóng kết nối</text>')
body.append(f'<text class="sub halo" x="{x(SILENCE) + 4:.1f}" y="{YA + BH + 16}">token từ giây {SILENCE} không tới</text>')

# Hàng B: ping mỗi 15 giây
body.append(f'<text class="row" x="20" y="{YB + 14}">Ping mỗi {PING} s</text>')
body.append(f'<text class="sub" x="20" y="{YB + 31}">im lặng tối đa {max(gaps)} s</text>')
body.append(f'<rect class="slot" x="{X0}" y="{YB}" width="{x(SILENCE) - X0:.1f}" height="{BH}" rx="5"/>')
for s in pings:
    body.append(f'<line class="lfg" x1="{x(s):.1f}" y1="{YB - 4}" x2="{x(s):.1f}" y2="{YB + BH + 4}" style="stroke-width:2.5"/>')
    body.append(f'<text class="sub mid halo" x="{x(s):.1f}" y="{YB - 9}" style="font-family:{svgkit.MONO}">: ping</text>')
body.append(f'<rect class="blue" x="{x(SILENCE):.1f}" y="{YB}" width="{X1 - x(SILENCE):.1f}" height="{BH}" rx="5"/>')
body.append(f'<text class="sub halo" x="{x(SILENCE) + 4:.1f}" y="{YB + BH + 16}">token từ giây {SILENCE}</text>')
# Ngoặc chỉ khoảng im lặng dài nhất
gx0, gx1 = x(0), x(PING)
body.append(f'<path class="lfg" d="M{gx0:.1f},{YB + BH + 8} v6 H{gx1:.1f} v-6"/>')
body.append(f'<text class="sub mid" x="{(gx0 + gx1) / 2:.1f}" y="{YB + BH + 28}">{PING} s &lt; {IDLE} s</text>')

# Trục thời gian
ay = YB + BH + 46
body.append(f'<line class="axis" x1="{X0}" y1="{ay}" x2="{X1}" y2="{ay}"/>')
for s in range(0, TEND + 1, 20):
    body.append(f'<line class="axis" x1="{x(s):.1f}" y1="{ay}" x2="{x(s):.1f}" y2="{ay + 5}"/>'
                f'<text class="sub mid" x="{x(s):.1f}" y="{ay + 18}">{s} s</text>')

body.append(f'<text class="cap2" x="20" y="{H - 12}">Giả định: im lặng {SILENCE} s trước token đầu; idle timeout '
            f'{IDLE} s (mặc định của nginx và AWS ALB).</text>')

svgkit.write(__file__, "idle-heartbeat.svg", svgkit.svg(
    W, H, f"Model im lặng {SILENCE} giây trong khi idle timeout là {IDLE} giây: không heartbeat thì kết nối bị đóng "
    f"ở giây {IDLE} và token không tới client; gửi ping mỗi {PING} giây thì khoảng im lặng dài nhất chỉ {PING} giây",
    body))
