"""Sinh graceful-shutdown.svg: SVG tĩnh, dòng thời gian khi một pod vLLM bị dừng (§2.4).

Số lấy từ manifest và các bước ở §2.4 (script đọc README để chắc còn khớp):
- preStop `sleep 15`, `--shutdown-timeout=90`, `terminationGracePeriodSeconds: 120`
  = 15 preStop + 90 drain + 15 dọn dẹp.
- Giây 0: pod Terminating; cùng lúc endpoint chuyển `ready: false`, các load
  balancer cập nhật mỗi nơi một lúc nên pod có thể còn nhận request mới.
- SIGTERM chỉ được gửi sau khi preStop xong (giây 15). vLLM ngừng nhận request
  mới, chạy nốt request đang có tối đa 90 giây, hết hạn thì huỷ phần còn lại
  (giây 105). Giây 120 (tính từ lúc preStop bắt đầu): SIGKILL.
Ba request minh hoạ có độ dài giả định: A đang chạy, xong ở giây 60; B đang chạy,
cần tới giây 115 nên bị huỷ ở giây 105; C đến ở giây 5 lúc load balancer chưa
cập nhật, vẫn được nhận vì chưa có SIGTERM, xong ở giây 40.
Hình tĩnh vì các mốc đọc được trên trục thời gian; chuyển động không thêm ý.

Chạy: python3 make_graceful_shutdown.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
PRESTOP, DRAIN, GRACE = 15, 90, 120
CLEANUP = GRACE - PRESTOP - DRAIN
assert CLEANUP == 15
assert f'command: ["sleep", "{PRESTOP}"]' in README
assert f"--shutdown-timeout={DRAIN}" in README
assert f"terminationGracePeriodSeconds: {GRACE}" in README
assert f"ở đây {GRACE} = {PRESTOP} `preStop` + {DRAIN} drain + {CLEANUP} dọn dẹp" in README
assert "SIGTERM chỉ được gửi sau khi `preStop` xong" in README
SIGTERM, CANCEL, SIGKILL = PRESTOP, PRESTOP + DRAIN, GRACE

REQ_A_END, REQ_B_END = 60, 115
REQ_C = (5, 40)
assert SIGTERM < REQ_A_END <= CANCEL < REQ_B_END            # A xong trong lúc drain, B bị huỷ
assert REQ_C[0] < SIGTERM and REQ_C[1] <= CANCEL            # C đến trước SIGTERM nên vẫn được chạy

W, H = 980, 470
X0, X1 = 200, 930
TMAX = 130
BH = 30


def x(s: float) -> float:
    return X0 + s / TMAX * (X1 - X0)


def bar(s0: float, s1: float, y: float, cls: str, label: str = "", lcls: str = "lbl") -> str:
    out = f'<rect class="{cls}" x="{x(s0) + 1:.1f}" y="{y}" width="{x(s1) - x(s0) - 2:.1f}" height="{BH}" rx="5"/>'
    if label:
        out += (f'<text class="{lcls}" style="font-size:12px" x="{(x(s0) + x(s1)) / 2:.1f}" '
                f'y="{y + BH / 2 + 5}">{label}</text>')
    return out


body: list[str] = []
body.append('<text class="title" x="20" y="38">Dừng một pod vLLM: preStop, drain, rồi mới SIGKILL</text>')
body.append(svgkit.legend(20, 70, [("tok", "chờ, dọn dẹp"), ("blue", "request đang chạy"),
                                    ("hatch", "request bị huỷ")]))

YE, YK, YA, YB, YC = 110, 160, 240, 285, 330
rows = [(YE, "Endpoint", "load balancer"), (YK, "Container", "kubelet, vLLM"),
        (YA, "Request A", "đang chạy"), (YB, "Request B", "đang chạy, dài"),
        (YC, "Request C", "đến ở giây 5")]
for y, name, sub in rows:
    body.append(f'<text class="row" x="20" y="{y + 13}">{name}</text>'
                f'<text class="sub" x="20" y="{y + 29}">{sub}</text>')

# Endpoint: ready:false ngay giây 0, các nơi gỡ pod mỗi nơi một lúc
body.append(f'<rect class="slot" x="{x(0) + 1:.1f}" y="{YE}" width="{x(PRESTOP) - x(0) - 2:.1f}" height="{BH}" rx="5"/>')
body.append(f'<text class="sub" x="{x(0) + 6:.1f}" y="{YE - 6}">ready: false; mỗi load balancer gỡ pod một lúc</text>')
body.append(f'<text class="sub" x="{x(PRESTOP) + 8:.1f}" y="{YE + BH / 2 + 5}">không còn nhận traffic mới</text>')

# Container: preStop -> drain -> dọn dẹp
body.append(bar(0, PRESTOP, YK, "tok", f"sleep {PRESTOP}"))
body.append(bar(SIGTERM, CANCEL, YK, "tok", f"--shutdown-timeout={DRAIN}: không nhận mới, chạy nốt"))
body.append(bar(CANCEL, SIGKILL, YK, "tok", "dọn"))

# Mốc dọc
for t in (SIGTERM, CANCEL, SIGKILL):
    xt = x(t)
    body.append(f'<line class="lfg dash" x1="{xt:.1f}" y1="{YK - 4}" x2="{xt:.1f}" y2="{YC + BH + 6}"/>')
dy = YK + BH + 16
body.append(f'<text class="sub halo" x="{x(SIGTERM) + 4:.1f}" y="{dy}">SIGTERM, giây {SIGTERM}</text>')
body.append(f'<text class="sub halo end" x="{x(CANCEL) - 4:.1f}" y="{dy}">hết {DRAIN} s: huỷ phần còn lại</text>')
body.append(f'<text class="sub halo" x="{x(SIGKILL) - 30:.1f}" y="{YK - 10}">SIGKILL</text>')

# Request
body.append(bar(0, REQ_A_END, YA, "blue", "xong trong lúc drain", "cap mid"))
body.append(bar(0, CANCEL, YB, "blue", "chạy tới hết hạn drain", "cap mid"))
body.append(bar(CANCEL, REQ_B_END, YB, "hatch"))
body.append(f'<text class="sub halo mid" x="{(x(CANCEL) + x(REQ_B_END)) / 2:.1f}" y="{YB - 5}">bị huỷ</text>')
body.append(bar(REQ_C[0], REQ_C[1], YC, "blue", "vẫn được nhận", "cap mid"))

# Trục thời gian
ay = YC + BH + 22
body.append(f'<line class="axis" x1="{X0}" y1="{ay}" x2="{X1}" y2="{ay}"/>')
for s in range(0, TMAX + 1, 15):
    body.append(f'<line class="axis" x1="{x(s):.1f}" y1="{ay}" x2="{x(s):.1f}" y2="{ay + 5}"/>'
                f'<text class="sub mid" x="{x(s):.1f}" y="{ay + 18}">{s} s</text>')

body.append(f'<text class="cap" x="20" y="{H - 36}" style="font-weight:600">terminationGracePeriodSeconds {GRACE} = '
            f'{PRESTOP} preStop + {DRAIN} drain + {CLEANUP} dọn dẹp</text>')
body.append(f'<text class="cap2" x="20" y="{H - 14}">Mặc định --shutdown-timeout 0: SIGTERM huỷ ngay mọi request đang '
            'chạy. Độ dài ba request là giả định.</text>')

assert X1 <= W - 30   # lề phải ≥ 30 px

ALT = (f"Khi pod bị dừng, preStop sleep {PRESTOP} giây cho load balancer kịp gỡ pod, rồi mới SIGTERM; vLLM chạy nốt "
       f"request đang có tối đa {DRAIN} giây rồi huỷ phần còn lại; SIGKILL ở giây {GRACE}. Request dài hơn hạn drain "
       "vẫn bị cắt")
assert f"![{ALT}](images/graceful-shutdown.svg)" in README   # alt trong bài khớp số trên hình
svgkit.write(__file__, "graceful-shutdown.svg", svgkit.svg(W, H, ALT, body))
